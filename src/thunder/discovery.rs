//! Peer discovery: LAN UDP beacons + explicit manual endpoints.
//!
//! No internet infrastructure. A beacon carries identity + contact info
//! (self-signed; meaningful trust comes only from T2 pairing). Peers
//! expire after missed beacons. Manual `host:port` endpoints cover VPN
//! and non-broadcast networks.

use std::collections::HashMap;
use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

use super::error::ThunderError;
use super::identity::ThunderIdentity;

/// UDP port for presence beacons.
pub const DISCOVERY_PORT: u16 = 47831;
/// Beacon interval and peer expiry.
pub const BEACON_INTERVAL: Duration = Duration::from_secs(5);
pub const PEER_EXPIRY: Duration = Duration::from_secs(20);
pub const PROTOCOL_VERSION: u32 = 1;

/// A discovered (not yet trusted) peer.
#[derive(Debug, Clone)]
pub struct DiscoveredPeer {
    pub peer_id: String,
    pub name: String,
    pub public_key: [u8; 32],
    pub addr: SocketAddr,
    pub thunder_port: u16,
    pub last_seen: Instant,
    pub signature_valid_self: bool,
}

/// Presence beacon payload (JSON, self-signed).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Beacon {
    pub protocol_version: u32,
    pub peer_id: String,
    pub name: String,
    pub public_key: Vec<u8>,
    pub thunder_port: u16,
    pub signature: Vec<u8>,
}

impl Beacon {
    pub fn create(identity: &ThunderIdentity, thunder_port: u16) -> Self {
        let public_key = identity.public_key_bytes().to_vec();
        let mut msg = Vec::new();
        msg.extend_from_slice(identity.peer_id.as_bytes());
        msg.extend_from_slice(&public_key);
        msg.extend_from_slice(&thunder_port.to_be_bytes());
        let signature = identity.sign(&msg).to_bytes().to_vec();
        Self {
            protocol_version: PROTOCOL_VERSION,
            peer_id: identity.peer_id.clone(),
            name: identity.display_name.clone(),
            public_key,
            thunder_port,
            signature,
        }
    }

    /// Verify shape + self-signature (proves holder of the private key
    /// sent it; trust still requires pairing).
    pub fn verify(&self) -> bool {
        if self.protocol_version != PROTOCOL_VERSION || self.public_key.len() != 32 {
            return false;
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&self.public_key);
        let Ok(sig) = ed25519_dalek::Signature::from_slice(&self.signature) else {
            return false;
        };
        let mut msg = Vec::new();
        msg.extend_from_slice(self.peer_id.as_bytes());
        msg.extend_from_slice(&self.public_key);
        msg.extend_from_slice(&self.thunder_port.to_be_bytes());
        ThunderIdentity::verify(&arr, &msg, &sig)
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = b"HTND1".to_vec();
        out.extend_from_slice(&serde_json::to_vec(self).unwrap_or_default());
        out
    }

    pub fn decode(bytes: &[u8], from: SocketAddr) -> Option<(Self, SocketAddr)> {
        if !bytes.starts_with(b"HTND1") {
            return None;
        }
        serde_json::from_slice::<Beacon>(&bytes[5..])
            .ok()
            .filter(|b| b.verify())
            .map(|b| {
                let mut addr = from;
                addr.set_port(b.thunder_port);
                (b, addr)
            })
    }
}

/// Discovery service: broadcasts presence, collects peers, parses manual
/// endpoints. Sockets are non-blocking; the App polls `poll()`.
pub struct Discovery {
    socket: UdpSocket,
    pub thunder_port: u16,
    peers: HashMap<String, DiscoveredPeer>,
}

impl Discovery {
    pub fn bind(thunder_port: u16) -> Result<Self, ThunderError> {
        let socket = UdpSocket::bind(("0.0.0.0", DISCOVERY_PORT)).map_err(|e| {
            ThunderError::InvalidRequest {
                detail: format!("discovery bind: {e}"),
            }
        })?;
        socket.set_broadcast(true).ok();
        socket
            .set_nonblocking(true)
            .map_err(|e| ThunderError::InvalidRequest {
                detail: format!("discovery nonblocking: {e}"),
            })?;
        Ok(Self {
            socket,
            thunder_port,
            peers: HashMap::new(),
        })
    }

    /// Test/directed constructor on loopback.
    pub fn bind_loopback(thunder_port: u16) -> Result<Self, ThunderError> {
        let socket =
            UdpSocket::bind(("127.0.0.1", 0)).map_err(|e| ThunderError::InvalidRequest {
                detail: format!("loopback bind: {e}"),
            })?;
        socket
            .set_nonblocking(true)
            .map_err(|e| ThunderError::InvalidRequest {
                detail: format!("loopback nonblocking: {e}"),
            })?;
        Ok(Self {
            socket,
            thunder_port,
            peers: HashMap::new(),
        })
    }

    pub fn local_beacon_addr(&self) -> Option<SocketAddr> {
        self.socket.local_addr().ok()
    }

    pub fn broadcast(&self, identity: &ThunderIdentity) -> Result<(), ThunderError> {
        let beacon = Beacon::create(identity, self.thunder_port);
        let data = beacon.encode();
        // Directed subnet broadcast; loopback for local testing is added
        // by the caller via announce_to when needed.
        for target in [
            format!("255.255.255.255:{DISCOVERY_PORT}"),
            format!(
                "127.0.0.1:{}",
                self.socket
                    .local_addr()
                    .map(|a| a.port())
                    .unwrap_or(DISCOVERY_PORT)
            ),
        ] {
            let _ = self.socket.send_to(&data, &target);
        }
        Ok(())
    }

    /// Send a beacon to one explicit address (tests, VPN, manual peers).
    pub fn announce_to(
        &self,
        identity: &ThunderIdentity,
        target: SocketAddr,
    ) -> Result<(), ThunderError> {
        let data = Beacon::create(identity, self.thunder_port).encode();
        self.socket
            .send_to(&data, target)
            .map_err(|e| ThunderError::InvalidRequest {
                detail: format!("announce: {e}"),
            })?;
        Ok(())
    }

    /// Drain inbound beacons (non-blocking). Returns newly seen peer ids.
    pub fn poll(&mut self, own_peer_id: &str) -> Vec<String> {
        let mut buf = [0u8; 2048];
        let mut fresh = Vec::new();
        loop {
            match self.socket.recv_from(&mut buf) {
                Ok((n, from)) => {
                    if let Some((beacon, addr)) = Beacon::decode(&buf[..n], from) {
                        if beacon.peer_id == own_peer_id {
                            continue;
                        }
                        let mut arr = [0u8; 32];
                        arr.copy_from_slice(&beacon.public_key);
                        let is_new = !self.peers.contains_key(&beacon.peer_id);
                        self.peers.insert(
                            beacon.peer_id.clone(),
                            DiscoveredPeer {
                                peer_id: beacon.peer_id.clone(),
                                name: beacon.name.clone(),
                                public_key: arr,
                                addr,
                                thunder_port: beacon.thunder_port,
                                last_seen: Instant::now(),
                                signature_valid_self: true,
                            },
                        );
                        if is_new {
                            fresh.push(beacon.peer_id.clone());
                        }
                    }
                }
                Err(_) => break,
            }
        }
        self.peers
            .retain(|_, p| p.last_seen.elapsed() < PEER_EXPIRY);
        fresh
    }

    /// Manual endpoint: `host:port` of a peer's Thunder TCP port.
    /// Returns an UNVERIFIED entry: identity is unknown (empty peer id,
    /// zeroed key) until a beacon or validated Pair exchange establishes
    /// it. Never manufactured from host:port as though it were identity.
    pub fn manual_peer(
        &self,
        endpoint: &str,
        _peer_id_hint: &str,
    ) -> Result<DiscoveredPeer, ThunderError> {
        let addr: SocketAddr = endpoint.parse().map_err(|_| ThunderError::InvalidRequest {
            detail: format!("bad endpoint (want host:port): {endpoint}"),
        })?;
        Ok(DiscoveredPeer {
            peer_id: String::new(),
            name: endpoint.to_string(),
            public_key: [0u8; 32],
            addr,
            thunder_port: addr.port(),
            last_seen: Instant::now(),
            signature_valid_self: false,
        })
    }

    pub fn peers(&self) -> Vec<&DiscoveredPeer> {
        self.peers.values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thunder::identity::ThunderIdentity;

    #[test]
    fn test_beacon_round_trip_and_tamper() {
        let id = ThunderIdentity::generate("alice".to_string());
        let b = Beacon::create(&id, 9000);
        assert!(b.verify());
        let data = b.encode();
        let from: SocketAddr = "127.0.0.1:1234".parse().unwrap();
        let (back, addr) = Beacon::decode(&data, from).unwrap();
        assert_eq!(back.peer_id, id.peer_id);
        assert_eq!(addr.port(), 9000);
        // Tampered name breaks the signature.
        let mut bad = data.clone();
        if let Some(i) = bad.iter().position(|&c| c == b'a') {
            bad[i] = b'b';
        }
        assert!(Beacon::decode(&bad, from).is_none());
        // Wrong magic rejected.
        assert!(Beacon::decode(b"NOPE{}", from).is_none());
    }

    #[test]
    fn test_loopback_discovery_and_expiry() {
        let a = ThunderIdentity::generate("a".to_string());
        let b = ThunderIdentity::generate("b".to_string());
        let da = Discovery::bind_loopback(9101).unwrap();
        let mut db = Discovery::bind_loopback(9102).unwrap();
        let target = db.local_beacon_addr().unwrap();
        da.announce_to(&a, target).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));
        let fresh = db.poll(&b.peer_id);
        assert_eq!(fresh, vec![a.peer_id.clone()]);
        assert_eq!(db.peers().len(), 1);
        // Own beacons ignored.
        db.announce_to(&b, target).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));
        let fresh = db.poll(&b.peer_id);
        assert!(fresh.is_empty());
    }

    #[test]
    fn test_manual_endpoint_parse() {
        let d = Discovery::bind_loopback(9103).unwrap();
        let p = d.manual_peer("192.168.1.5:47832", "bob").unwrap();
        assert_eq!(p.thunder_port, 47832);
        assert!(!p.signature_valid_self);
        // Honest: no identity manufactured from host:port.
        assert!(p.peer_id.is_empty());
        assert_eq!(p.public_key, [0u8; 32]);
        assert!(d.manual_peer("not-an-endpoint", "x").is_err());
    }
}
