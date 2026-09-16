//! Peer health tracking: online/offline/busy/degraded with latency,
//! failure counts and active requests. A failed host degrades locally
//! and never poisons the agent session.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Health of one peer from the receiver's point of view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerHealth {
    Online,
    Offline,
    Connecting,
    Busy,
    Degraded,
}

#[derive(Debug, Clone)]
pub struct PeerStats {
    pub health: PeerHealth,
    pub last_seen: Option<Instant>,
    pub rtt_ms: Option<u64>,
    pub time_to_first_token_ms: Option<u64>,
    pub tokens_per_sec: Option<f64>,
    pub failure_count: u32,
    pub active_requests: u32,
}

impl Default for PeerStats {
    fn default() -> Self {
        Self {
            health: PeerHealth::Offline,
            last_seen: None,
            rtt_ms: None,
            time_to_first_token_ms: None,
            tokens_per_sec: None,
            failure_count: 0,
            active_requests: 0,
        }
    }
}

/// Per-peer health table.
#[derive(Debug, Default)]
pub struct PeerHealthTable {
    stats: HashMap<String, PeerStats>,
}

impl PeerHealthTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stats(&mut self, peer_id: &str) -> &mut PeerStats {
        self.stats.entry(peer_id.to_string()).or_default()
    }

    pub fn get(&self, peer_id: &str) -> Option<&PeerStats> {
        self.stats.get(peer_id)
    }

    /// Record a successful round trip; decay failure count.
    pub fn success(&mut self, peer_id: &str, rtt: Duration) {
        let s = self.stats(peer_id);
        s.last_seen = Some(Instant::now());
        s.rtt_ms = Some(rtt.as_millis() as u64);
        s.failure_count = s.failure_count.saturating_sub(1);
        if s.health == PeerHealth::Offline || s.health == PeerHealth::Connecting {
            s.health = PeerHealth::Online;
        }
    }

    /// Record a failed request. Three consecutive failures degrade;
    /// recovery happens one success at a time via `success()`.
    pub fn failure(&mut self, peer_id: &str) {
        let s = self.stats(peer_id);
        s.failure_count += 1;
        if s.failure_count >= 3 {
            s.health = PeerHealth::Degraded;
        }
        if s.failure_count >= 10 {
            s.health = PeerHealth::Offline;
        }
    }

    pub fn request_started(&mut self, peer_id: &str) {
        self.stats(peer_id).active_requests += 1;
    }

    pub fn request_finished(&mut self, peer_id: &str) {
        let s = self.stats(peer_id);
        s.active_requests = s.active_requests.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_degrade_and_recover() {
        let mut t = PeerHealthTable::new();
        t.success("a", Duration::from_millis(40));
        assert_eq!(t.get("a").unwrap().health, PeerHealth::Online);
        t.failure("a");
        t.failure("a");
        assert_eq!(t.get("a").unwrap().health, PeerHealth::Online);
        t.failure("a");
        assert_eq!(t.get("a").unwrap().health, PeerHealth::Degraded);
        // One success doesn't instantly clear degraded (decay by one…
        t.success("a", Duration::from_millis(40));
        assert_eq!(t.get("a").unwrap().failure_count, 2);
    }

    #[test]
    fn test_active_request_counting() {
        let mut t = PeerHealthTable::new();
        t.request_started("a");
        t.request_started("a");
        t.request_finished("a");
        assert_eq!(t.get("a").unwrap().active_requests, 1);
        t.request_finished("a");
        t.request_finished("a");
        assert_eq!(t.get("a").unwrap().active_requests, 0);
    }
}
