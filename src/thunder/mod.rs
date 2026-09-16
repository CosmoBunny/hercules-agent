//! Shared Thunder: decentralized encrypted P2P inference (T1+).
//!
//! Boundary (non-negotiable): inference sharing only — never filesystem,
//! shell, tools, or credentials. The receiver owns context; the host owns
//! compute. Phase order: T1 identity → T2 pairing/encryption → … → T13.

pub mod capabilities;
pub mod connection;
pub mod context;
pub mod crypto;
pub mod discovery;
pub mod error;
pub mod host;
pub mod identity;
pub mod pairing;
pub mod peer;
pub mod protocol;
pub mod receiver;
pub mod routing;
pub mod runtime;
pub mod swarm;

pub use capabilities::{HostCapabilities, ThunderModel};
pub use discovery::{DISCOVERY_PORT, DiscoveredPeer, Discovery};
pub use error::ThunderError;
pub use identity::ThunderIdentity;
pub use pairing::{PairingCode, PeerPermissions, PeerStore, TrustedPeer};

pub(crate) fn identity_path() -> std::path::PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("hercules")
        .join("thunder")
}
