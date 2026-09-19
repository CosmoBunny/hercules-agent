//! Shared Thunder TUI integration state (transient UI only).
//!
//! Boundary: this module owns NO persistent, trust, network, or model
//! state. Identity comes from `ThunderIdentity`, trust from `PeerStore`,
//! discovery from `Discovery`, execution from `ThunderHost` /
//! `AgentBackendExecutor`, remote inference from `SharedThunderBackend`
//! / `ThunderReceiver`. This module only holds transient UI state
//! (selected tab/peer/model, input buffers, pending pairing, host
//! operation handles) and orchestrates the existing components.
//!
//! Honesty rules (no dummy production behavior):
//! - The host serves exactly ONE real model: the active local backend.
//!   Advertised id == executed id (checked in `AgentBackendExecutor`).
//! - Pairing is a real exchange: the receiver sends `Pair` with the
//!   out-of-band code; the host validates it against the active secret
//!   and queues the attempt; only explicit operator Trust persists.
//! - `verified=true` is set only after a validated Pair exchange with
//!   the fingerprint shown plus explicit confirmation.
//! - Endpoints come from the real listener (`local_addr`); discovery
//!   advertises that exact TCP port; remote addresses come from beacon
//!   source IP + advertised thunder port — never manufactured.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use crate::thunder::capabilities::{HostCapabilities, ThunderModel};
use crate::thunder::discovery::DiscoveredPeer;
use crate::thunder::error::ThunderError;
use crate::thunder::host::PairingRegistry;
use crate::thunder::identity::ThunderIdentity;
use crate::thunder::pairing::{PairingCode, PeerStore, TrustedPeer};
use crate::thunder::runtime::RemoteTarget;

/// Thunder modal internal tabs: Overview | Host | Connect | Peers | Models | Settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThunderTab {
    Overview,
    Host,
    Connect,
    Peers,
    Models,
    Settings,
}

impl ThunderTab {
    pub const ALL: [ThunderTab; 6] = [
        ThunderTab::Overview,
        ThunderTab::Host,
        ThunderTab::Connect,
        ThunderTab::Peers,
        ThunderTab::Models,
        ThunderTab::Settings,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ThunderTab::Overview => "Overview",
            ThunderTab::Host => "Host",
            ThunderTab::Connect => "Connect",
            ThunderTab::Peers => "Peers",
            ThunderTab::Models => "Models",
            ThunderTab::Settings => "Settings",
        }
    }

    pub fn from_index(i: usize) -> ThunderTab {
        ThunderTab::ALL[i.min(ThunderTab::ALL.len() - 1)]
    }

    pub fn index(self) -> usize {
        ThunderTab::ALL.iter().position(|t| *t == self).unwrap_or(0)
    }
}

/// One inbound connection attempt observed by the host accept loop
/// (handshake identity only — NOT a validated pairing attempt).
#[derive(Debug, Clone)]
pub struct InboundPeer {
    pub peer_id: String,
    pub name: String,
    pub first_seen_epoch: u64,
}

/// One remote model entry: exact peer + address + model + the trusted
/// record for THIS peer. Selecting it yields the atomic `RemoteTarget`
/// — never a peer reconstructed from an arbitrary store entry.
#[derive(Debug, Clone)]
pub struct RemoteModelEntry {
    pub peer_id: String,
    pub peer_name: String,
    pub address: SocketAddr,
    pub model: ThunderModel,
    pub trusted: TrustedPeer,
}

impl RemoteModelEntry {
    /// Display label for the Main AI selector: explicit remote target.
    pub fn selector_label(&self) -> String {
        format!("Thunder / {} / {}", self.peer_name, self.model.name)
    }

    /// Selection reference: `peer_id::model_id@host:port`.
    pub fn model_ref(&self) -> String {
        remote_model_ref(&self.peer_id, &self.model.id, self.address)
    }

    /// The exact atomic target — peer identity, endpoint and model
    /// travel together from selection to runtime.
    pub fn target(&self) -> RemoteTarget {
        make_remote_target(
            &self.peer_id,
            self.address,
            &self.model.id,
            self.trusted.clone(),
        )
    }
}

/// Text input focus inside the Thunder modal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThunderInputMode {
    #[default]
    None,
    ManualEndpoint,
    PairingCode,
}

/// Host identity as established by a validated Pair exchange: the host
/// accepted our code and presented this identity over the encrypted
/// channel. The operator confirms the fingerprint before Trust.
#[derive(Debug, Clone)]
pub struct AcceptedHost {
    pub peer_id: String,
    pub name: String,
    pub public_key: [u8; 32],
}

/// Receiver-side pending pairing: the operator selected a peer address
/// and submitted the out-of-band code. `accepted` is Some ONLY after
/// the host validated the code and replied with its identity. Nothing
/// is trusted until the operator explicitly confirms with the peer
/// fingerprint visible.
#[derive(Debug, Clone)]
pub struct PendingPairing {
    pub address: SocketAddr,
    pub code_entered: String,
    pub accepted: Option<AcceptedHost>,
}

/// Last-known exchange health per peer id (transient). A beacon only
/// proves recent presence advertisements — NOT a healthy TCP path.
/// Online/offline derive from actual authenticated exchanges.
#[derive(Debug, Clone, Default)]
pub struct PeerHealth {
    pub last_ok_epoch: Option<u64>,
    pub last_err: Option<String>,
    pub last_err_epoch: Option<u64>,
}

impl PeerHealth {
    /// Honest one-line state: trust + freshness + exchange outcome.
    /// Never "Available"/"Connected" from presence alone.
    pub fn label(&self, trusted: bool, fresh: bool) -> &'static str {
        if trusted {
            if self.last_ok_epoch.is_some() && self.last_ok_epoch > self.last_err_epoch {
                "trusted·online"
            } else if self.last_err.is_some() {
                "trusted·unreachable"
            } else {
                "trusted"
            }
        } else if fresh {
            "discovered·unverified"
        } else {
            "stale·unverified"
        }
    }

    pub fn record_ok(&mut self) {
        let now = epoch_now();
        self.last_ok_epoch = Some(now);
        self.last_err = None;
        self.last_err_epoch = None;
    }

    pub fn record_err(&mut self, err: String) {
        self.last_err_epoch = Some(epoch_now());
        self.last_err = Some(err);
    }
}

fn epoch_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The live host runtime: ONE authoritative TCP port owned by the real
/// listener. The endpoint, the discovery advertisement, and the UI
/// display all derive from this single value.
pub struct ThunderHostRuntime {
    /// Display endpoint: selected usable interface IP + actual bound port.
    /// When no usable LAN interface exists this is loopback + port and
    /// `lan_available` is false — the UI must then show "loopback only",
    /// never present it as a LAN endpoint.
    pub endpoint: SocketAddr,
    /// Authoritative Thunder TCP port (== endpoint.port()).
    pub tcp_port: u16,
    /// Interface the display endpoint was taken from ("loopback-only"
    /// when no usable interface exists).
    pub iface_name: String,
    /// Every usable interface IP + bound port (deterministic order).
    /// Empty when no usable LAN interface exists.
    pub all_endpoints: Vec<SocketAddr>,
    /// False when no non-loopback interface exists: the endpoint is
    /// loopback-only and NOT LAN-reachable.
    pub lan_available: bool,
    /// Wire model id actually served (== executed id).
    pub model_id: String,
    /// Real backend model name for display.
    pub model_name: String,
    /// Real backend label (e.g. "llama.cpp (foo.gguf)").
    pub backend_label: String,
    /// Host-wide shutdown: cancels the accept loop AND every live
    /// connection task (sessions close, active generation aborts).
    pub stop_token: tokio_util::sync::CancellationToken,
    pub registry: Arc<PairingRegistry>,
    pub inbound: Arc<Mutex<Vec<InboundPeer>>>,
}

/// The ONE real model this host can execute: the active local backend.
/// `wire_id` is the validated wire identity (== what Generate must
/// carry); everything else is genuine backend metadata or an explicit
/// default bound (never fabricated capabilities).
pub struct HostedModel {
    pub wire_id: String,
    pub display_name: String,
    pub architecture: String,
    pub format: String,
    pub backend_label: String,
    pub context_tokens: u32,
    /// "backend" when the backend reported its limit, else the explicit
    /// enforcement default (shown as such in the UI).
    pub context_source: &'static str,
    pub streaming: bool,
    pub cancellation: bool,
}

/// Enforcement default when the backend reports no context limit.
/// A policy bound the host actually enforces — NOT a model capability
/// claim (the UI labels it as a default bound).
pub const DEFAULT_CTX_BOUND: u32 = 8192;

/// Wire-safe model id (ThunderModel charset: alphanumeric, `-`, `_`).
/// Returns None when nothing usable remains — such backends refuse to
/// host rather than advertise a fabricated id.
pub fn sanitize_wire_id(raw: &str) -> Option<String> {
    let mut clean = String::new();
    let mut last_dash = false;
    for c in raw.trim().chars() {
        if c.is_ascii_alphanumeric() {
            clean.push(c.to_ascii_lowercase());
            last_dash = false;
        } else if matches!(c, '-' | '_') {
            if !last_dash {
                clean.push(c);
            }
            last_dash = true;
        } else if !last_dash && !clean.is_empty() {
            clean.push('-');
            last_dash = true;
        }
    }
    let clean = clean.trim_matches('-').to_string();
    if clean.is_empty()
        || clean
            .bytes()
            .any(|b| !(b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_')))
    {
        return None;
    }
    Some(clean)
}

/// Describe the ACTUAL active backend as the single hosted model.
/// Fails honestly (refuse to host) when the backend has no servable
/// local model — e.g. a SharedThunder backend cannot re-host a remote.
///
/// Honesty rules: `architecture` is "unknown" unless genuinely sourced
/// (Transformers `expected_arch`, which comes from the configured
/// model); `quantization` is always None (no backend exposes it —
/// never guessed from filenames); `format`/`backend_label` come from
/// the real runtime; `context_tokens` is the backend limit when
/// reported, else the enforced `DEFAULT_CTX_BOUND` policy (labeled via
/// `context_source`, never presented as a native capability).
pub fn host_model_from_backend(
    backend: &crate::backend::AgentBackend,
) -> Result<HostedModel, String> {
    use crate::backend::AgentBackend as B;
    match backend {
        B::Ollama(b) => {
            let wire_id = sanitize_wire_id(&b.model)
                .ok_or_else(|| format!("Ollama model name {:?} has no wire-safe id", b.model))?;
            Ok(HostedModel {
                wire_id,
                display_name: b.model.clone(),
                architecture: "unknown".to_string(),
                format: "Ollama".to_string(),
                backend_label: format!("Ollama ({})", b.model),
                context_tokens: DEFAULT_CTX_BOUND,
                context_source: "default bound",
                streaming: true,
                cancellation: true,
            })
        }
        B::LlamaCppLib(b) => {
            let rt = &b.runtime;
            let (display_name, format) = if let Some(ref p) = rt.model_path {
                let file = p
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| p.display().to_string());
                let fmt = if p.extension().map(|e| e == "gguf").unwrap_or(false) {
                    "GGUF"
                } else {
                    "llama.cpp"
                };
                (file, fmt.to_string())
            } else if !rt.model_name.is_empty() {
                (rt.model_name.clone(), "llama.cpp".to_string())
            } else {
                (rt.endpoint.clone(), "HTTP".to_string())
            };
            let wire_id = sanitize_wire_id(&display_name)
                .ok_or_else(|| format!("llama.cpp model {display_name:?} has no wire-safe id"))?;
            let (context_tokens, context_source) = match b.actual_context_limit() {
                Some(n) => (n.min(u32::MAX as usize) as u32, "backend"),
                None => (DEFAULT_CTX_BOUND, "default bound"),
            };
            Ok(HostedModel {
                wire_id,
                display_name,
                architecture: "unknown".to_string(),
                format,
                backend_label: b.name(),
                context_tokens: context_tokens.max(1),
                context_source,
                streaming: true,
                cancellation: true,
            })
        }
        B::Transformers(b) => {
            let dir_name = b
                .model_dir
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| b.model_dir.display().to_string());
            let wire_id = sanitize_wire_id(&dir_name)
                .ok_or_else(|| format!("Transformers dir {dir_name:?} has no wire-safe id"))?;
            Ok(HostedModel {
                wire_id,
                display_name: dir_name,
                architecture: b
                    .expected_arch
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                format: "SafeTensors".to_string(),
                backend_label: b.name(),
                context_tokens: DEFAULT_CTX_BOUND,
                context_source: "default bound",
                streaming: true,
                cancellation: true,
            })
        }
        #[cfg(feature = "gpu")]
        B::BurnWgpu(b) => {
            let wire_id = sanitize_wire_id(&b.model_name)
                .ok_or_else(|| format!("WGPU model {:?} has no wire-safe id", b.model_name))?;
            Ok(HostedModel {
                wire_id,
                display_name: b.model_name.clone(),
                architecture: "unknown".to_string(),
                format: "WGPU".to_string(),
                backend_label: backend.name(),
                context_tokens: DEFAULT_CTX_BOUND,
                context_source: "default bound",
                streaming: true,
                cancellation: true,
            })
        }
        B::SharedThunder(_) => Err(
            "cannot re-host a Thunder remote model (forwarding has no production path)".to_string(),
        ),
    }
}

impl HostedModel {
    /// The exact wire record: id + genuine metadata. `context_length`
    /// is the EFFECTIVE serving limit (backend limit or enforced
    /// default bound) — remote clients must treat it as host policy,
    /// not as the model's native context length. Unknown capabilities
    /// stay "unknown"/None, never fabricated.
    pub fn thunder_model(&self) -> ThunderModel {
        ThunderModel {
            id: self.wire_id.clone(),
            name: self.display_name.chars().take(48).collect(),
            architecture: self.architecture.clone(),
            format: self.format.clone(),
            backend: self.backend_label.chars().take(32).collect(),
            quantization: None,
            context_length: self.context_tokens.max(1),
            streaming: self.streaming,
            cancellation: self.cancellation,
        }
    }
}

/// Transient Thunder UI state. Persistent/trust/network/model state
/// always comes from the existing Thunder modules.
pub struct ThunderUiState {
    pub tab: usize,
    pub list_selected: usize,
    pub discovered: Vec<DiscoveredPeer>,
    /// Manually entered endpoints (unverified addresses — identity is
    /// established ONLY via beacon or validated Pair exchange).
    pub manual_peers: Vec<DiscoveredPeer>,
    /// Last-known TCP address per peer id (beacon source IP +
    /// advertised thunder port; manual entries use the typed endpoint).
    pub peer_addrs: HashMap<String, SocketAddr>,
    /// Exchange health per peer id (transient).
    pub peer_health: HashMap<String, PeerHealth>,
    pub manual_endpoint: String,
    pub code_input: String,
    pub input_mode: ThunderInputMode,
    pub host_runtime: Option<ThunderHostRuntime>,
    /// Configured host TCP port (0 = ephemeral). The authoritative port
    /// is always the bound listener's actual port.
    pub host_port: u16,
    pub allow_inference: bool,
    pub max_concurrent: u32,
    pub remote_models: Vec<RemoteModelEntry>,
    pub pending_pairing: Option<PendingPairing>,
    pub selected_remote: Option<RemoteTarget>,
    /// Peer chosen on the Connect tab for code submission (peer id, or
    /// `manual:<addr>` for identity-unknown manual entries).
    pub connect_peer: Option<String>,
    pub status: String,
    /// Internal tab bar hits: (tab_idx, row, x0, x1).
    pub tab_hits: Vec<(usize, u16, u16, u16)>,
    /// Content action hits: (action_idx, row).
    pub row_hits: Vec<(usize, u16)>,
    pub last_broadcast: Option<std::time::Instant>,
}

impl Default for ThunderUiState {
    fn default() -> Self {
        Self {
            tab: 0,
            list_selected: 0,
            discovered: Vec::new(),
            manual_peers: Vec::new(),
            peer_addrs: HashMap::new(),
            peer_health: HashMap::new(),
            manual_endpoint: String::new(),
            code_input: String::new(),
            input_mode: ThunderInputMode::None,
            host_runtime: None,
            host_port: 0,
            allow_inference: true,
            max_concurrent: 4,
            remote_models: Vec::new(),
            pending_pairing: None,
            selected_remote: None,
            connect_peer: None,
            status: String::new(),
            tab_hits: Vec::new(),
            row_hits: Vec::new(),
            last_broadcast: None,
        }
    }
}

impl ThunderUiState {
    pub fn tab(&self) -> ThunderTab {
        ThunderTab::from_index(self.tab)
    }

    pub fn set_tab(&mut self, tab: usize) {
        self.tab = tab.min(ThunderTab::ALL.len() - 1);
        self.list_selected = 0;
        self.input_mode = ThunderInputMode::None;
    }

    /// The host is live ONLY while the runtime exists (the TCP listener
    /// is alive exactly as long as its accept task runs).
    pub fn host_running(&self) -> bool {
        self.host_runtime.is_some()
    }

    /// Read-only snapshot of the live endpoint for rendering.
    pub fn host_endpoint(&self) -> Option<SocketAddr> {
        self.host_runtime.as_ref().map(|r| r.endpoint)
    }

    /// Local identity display triple: (display name, peer id, fingerprint).
    pub fn local_identity() -> (String, String, String) {
        match ThunderIdentity::load_or_generate("Hercules".to_string()) {
            Ok(id) => (
                id.display_name.clone(),
                id.peer_id.clone(),
                id.fingerprint(),
            ),
            Err(_) => (
                "Hercules".to_string(),
                "unavailable".to_string(),
                "--".to_string(),
            ),
        }
    }

    /// Valid (non-expired) pairing code text, if the live registry holds one.
    pub fn active_code(&self) -> Option<String> {
        self.host_runtime
            .as_ref()
            .and_then(|r| r.registry.active_code_text())
    }

    /// Validated pairing attempts awaiting operator Trust/Reject.
    pub fn pending_requests(&self) -> Vec<crate::thunder::host::PendingPairingRequest> {
        self.host_runtime
            .as_ref()
            .map(|r| r.registry.pending())
            .unwrap_or_default()
    }

    /// Snapshot of raw inbound (handshake-level) attempts for display.
    pub fn inbound_peers(&self) -> Vec<InboundPeer> {
        self.host_runtime
            .as_ref()
            .and_then(|r| r.inbound.lock().ok().map(|v| v.clone()))
            .unwrap_or_default()
    }

    /// Selection key for a peer row: real peer id, or `manual:<addr>`
    /// while identity is still unknown (never presented as identity).
    pub fn peer_key(peer_id: &str, addr: SocketAddr) -> String {
        if peer_id.is_empty() {
            format!("manual:{addr}")
        } else {
            peer_id.to_string()
        }
    }

    /// Real TCP dial address for a peer: beacon source IP + advertised
    /// Thunder TCP port. (The beacon source UDP port is ephemeral and
    /// must never be dialed; manual entries carry the typed port in
    /// `thunder_port`.)
    pub fn tcp_addr(p: &DiscoveredPeer) -> SocketAddr {
        SocketAddr::new(p.addr.ip(), p.thunder_port)
    }

    /// Persist explicit trust for one peer. `verified` must be true ONLY
    /// after a validated Pair exchange with the fingerprint shown plus
    /// explicit operator confirmation — callers enforce this.
    pub fn trust_peer(
        peer_id: &str,
        public_key: [u8; 32],
        name: &str,
        inference: bool,
        forwarding: bool,
        verified: bool,
    ) -> bool {
        let mut store = PeerStore::load();
        let ok = store.trust(
            peer_id.to_string(),
            public_key,
            name.to_string(),
            crate::thunder::pairing::PeerPermissions {
                inference,
                forwarding,
            },
            verified,
        );
        if ok {
            // PeerStore::save does not create the data dir; a fresh
            // installation must still persist trust.
            let _ = std::fs::create_dir_all(crate::thunder::identity_path());
            let _ = store.save();
        }
        ok
    }

    /// Remove trust for one peer (persisted).
    pub fn untrust_peer(peer_id: &str) -> bool {
        let mut store = PeerStore::load();
        let ok = store.reject(peer_id);
        if ok {
            let _ = std::fs::create_dir_all(crate::thunder::identity_path());
            let _ = store.save();
        }
        ok
    }

    /// Adjust persisted permissions for one already-trusted peer.
    pub fn set_peer_permissions(peer_id: &str, inference: bool, forwarding: bool) -> bool {
        let mut store = PeerStore::load();
        let ok = store.set_permissions(
            peer_id,
            crate::thunder::pairing::PeerPermissions {
                inference,
                forwarding,
            },
        );
        if ok {
            let _ = std::fs::create_dir_all(crate::thunder::identity_path());
            let _ = store.save();
        }
        ok
    }
}

/// Selection reference: `peer_id::model_id@host:port`. Resolved later
/// by `RemoteTarget::parse`, which binds the EXACT trusted peer by id.
pub fn remote_model_ref(peer_id: &str, model_id: &str, addr: SocketAddr) -> String {
    format!("{peer_id}::{model_id}@{addr}")
}

/// The exact atomic remote target: peer identity, endpoint and model
/// travel together. Never decomposed + recombined with an arbitrary
/// peer.
pub fn make_remote_target(
    peer_id: &str,
    address: SocketAddr,
    model_id: &str,
    trusted_peer: TrustedPeer,
) -> RemoteTarget {
    RemoteTarget {
        peer_id: peer_id.to_string(),
        address,
        model_id: model_id.to_string(),
        trusted_peer,
    }
}

/// Current Main AI target label: explicit local backend, explicit
/// Thunder remote, never ambiguous.
pub fn main_ai_label(backend: &crate::backend::AgentBackend) -> String {
    match backend {
        crate::backend::AgentBackend::SharedThunder(b) => {
            let peer_name = if b.target.trusted_peer.name.is_empty() {
                b.target.peer_id.clone()
            } else {
                b.target.trusted_peer.name.clone()
            };
            format!("Thunder / {} / {}", peer_name, b.target.model_id)
        }
        other => format!("Local ({})", other.name()),
    }
}

/// Build host capabilities for the single real hosted model.
/// Inference-only: no filesystem, shell, tool, or credential fields
/// exist on the wire type. Forwarding is never advertised (no
/// production forwarder exists).
pub fn build_host_caps(
    identity: &ThunderIdentity,
    hosted: &HostedModel,
    inference: bool,
    max_concurrent: u32,
) -> HostCapabilities {
    HostCapabilities {
        peer_id: identity.peer_id.clone(),
        hardware: crate::model::HardwareInfo::detect(),
        models: vec![hosted.thunder_model()],
        inference,
        forwarding: false,
        max_concurrent_requests: max_concurrent.clamp(1, 64),
        max_context_tokens: hosted.context_tokens.max(1),
        max_new_tokens: crate::thunder::runtime::REMOTE_MAX_NEW_TOKENS,
        streaming: hosted.streaming,
        cancellation: hosted.cancellation,
    }
}

/// One usable local interface address from real OS enumeration
/// (`getifaddrs`): interface name + IP. Loopback and unspecified
/// addresses are excluded at collection time — never fabricated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IfaceAddr {
    pub name: String,
    pub ip: std::net::IpAddr,
}

/// Enumerate local interface addresses via `getifaddrs`, excluding
/// loopback interfaces/addresses and unspecified addresses. Order is
/// the OS order (callers sort deterministically).
#[cfg(unix)]
pub fn local_interface_addrs() -> Vec<IfaceAddr> {
    let mut out = Vec::new();
    // SAFETY: getifaddrs/freeifaddrs pairing; we only read fields.
    unsafe {
        let mut head: *mut libc::ifaddrs = std::ptr::null_mut();
        if libc::getifaddrs(&mut head) != 0 || head.is_null() {
            return out;
        }
        let mut cur = head;
        while !cur.is_null() {
            let ifa = &*cur;
            let flags = ifa.ifa_flags as u32;
            let is_loopback = (flags & libc::IFF_LOOPBACK as u32) != 0;
            if !is_loopback && !ifa.ifa_addr.is_null() {
                let fam = (*ifa.ifa_addr).sa_family as i32;
                let ip = match fam {
                    libc::AF_INET => {
                        let sin = &*ifa.ifa_addr.cast::<libc::sockaddr_in>();
                        Some(std::net::IpAddr::from(std::net::Ipv4Addr::from(
                            u32::from_be(sin.sin_addr.s_addr),
                        )))
                    }
                    libc::AF_INET6 => {
                        let sin6 = &*ifa.ifa_addr.cast::<libc::sockaddr_in6>();
                        Some(std::net::IpAddr::from(sin6.sin6_addr.s6_addr))
                    }
                    _ => None,
                };
                if let Some(ip) = ip {
                    if !ip.is_loopback() && !ip.is_unspecified() {
                        let name = if ifa.ifa_name.is_null() {
                            String::new()
                        } else {
                            std::ffi::CStr::from_ptr(ifa.ifa_name)
                                .to_string_lossy()
                                .into_owned()
                        };
                        out.push(IfaceAddr { name, ip });
                    }
                }
            }
            cur = (*cur).ifa_next;
        }
        libc::freeifaddrs(head);
    }
    out
}

/// Non-Unix fallback: no `getifaddrs` available. Returns no addresses
/// rather than fabricating any — callers already handle the empty case
/// (no LAN row shown). A GetAdaptersAddresses implementation can replace
/// this when Windows CI exists to verify it.
#[cfg(not(unix))]
pub fn local_interface_addrs() -> Vec<IfaceAddr> {
    Vec::new()
}

/// Rank for deterministic preference: private LAN IPv4 first, then
/// other global IPv4, then IPv6 (unique-local/global), then anything
/// else (link-local etc. last — still real, just deprioritized).
fn iface_rank(ip: &std::net::IpAddr) -> u8 {
    match ip {
        std::net::IpAddr::V4(v4) => {
            let o = v4.octets();
            if o[0] == 10
                || (o[0] == 172 && (16..32).contains(&o[1]))
                || (o[0] == 192 && o[1] == 168)
            {
                0
            } else if v4.is_link_local() || v4.is_multicast() || v4.is_broadcast() {
                3
            } else {
                1
            }
        }
        std::net::IpAddr::V6(v6) => {
            let s = v6.segments();
            if (s[0] & 0xfe00) == 0xfc00 || (s[0] & 0xffc0) == 0xfe80 {
                3
            } else {
                2
            }
        }
    }
}

/// Deterministically sort interface addresses: (preference rank,
/// numeric IP, interface name). Pure function — unit-tested with
/// synthetic inputs (no network needed).
pub fn sort_iface_addrs(addrs: &mut [IfaceAddr]) {
    addrs.sort_by(|a, b| {
        (iface_rank(&a.ip), ip_sort_key(&a.ip), &a.name).cmp(&(
            iface_rank(&b.ip),
            ip_sort_key(&b.ip),
            &b.name,
        ))
    });
}

fn ip_sort_key(ip: &std::net::IpAddr) -> Vec<u8> {
    match ip {
        std::net::IpAddr::V4(v4) => v4.octets().to_vec(),
        std::net::IpAddr::V6(v6) => v6.octets().to_vec(),
    }
}

/// Select the single display address: best usable interface per
/// `sort_iface_addrs`, or None when no usable interface exists.
/// NEVER returns loopback — the caller must show "loopback only",
/// never label loopback as LAN.
pub fn select_host_address() -> Option<IfaceAddr> {
    let mut addrs = local_interface_addrs();
    sort_iface_addrs(&mut addrs);
    addrs.into_iter().next()
}

/// All usable endpoints for a bound port, in deterministic order.
/// Empty when no usable LAN interface exists.
pub fn all_host_endpoints(port: u16) -> Vec<SocketAddr> {
    let mut addrs = local_interface_addrs();
    sort_iface_addrs(&mut addrs);
    addrs
        .into_iter()
        .map(|a| SocketAddr::new(a.ip, port))
        .collect()
}

/// Start the production host on the configured port (0 = ephemeral):
/// real `ThunderListener` on all interfaces, real handshake, real
/// `ThunderHost::serve` with `AgentBackendExecutor` bound to the ONE
/// advertised model. The returned endpoint derives from the ACTUAL
/// listener address — never manufactured. Trust is re-loaded from
/// `PeerStore` per connection so operator Trust applies without
/// restart.
pub async fn start_host_task(
    backend: crate::backend::AgentBackend,
    caps: HostCapabilities,
    identity: ThunderIdentity,
    configured_port: u16,
) -> Result<ThunderHostRuntime, ThunderError> {
    use crate::thunder::connection::{ThunderConnection, ThunderListener};
    use crate::thunder::host::{AgentBackendExecutor, ThunderHost};
    let model_id =
        caps.models
            .first()
            .map(|m| m.id.clone())
            .ok_or_else(|| ThunderError::InvalidRequest {
                detail: "no hosted model".to_string(),
            })?;
    let mut host = ThunderHost::new(
        caps,
        Arc::new(AgentBackendExecutor {
            backend,
            advertised_model: model_id.clone(),
        }),
    )?;
    let registry = PairingRegistry::new();
    registry.set_host_identity(
        identity.peer_id.clone(),
        identity.display_name.clone(),
        identity.public_key_bytes(),
    );
    host.set_pairing_registry(registry.clone());
    let (listener, _) = ThunderListener::bind(configured_port).await?;
    let bound = listener.local_addr()?;
    // The listener is 0.0.0.0 (all interfaces): the display endpoint is
    // the best usable interface IP + the ACTUAL bound port — one
    // authoritative value owned by this runtime. When no usable
    // interface exists the endpoint is loopback-only and flagged as
    // NOT LAN-reachable (never mislabeled).
    let all_endpoints = all_host_endpoints(bound.port());
    let (endpoint, iface_name, lan_available) = match select_host_address() {
        Some(sel) => (SocketAddr::new(sel.ip, bound.port()), sel.name, true),
        None => (
            SocketAddr::new(std::net::IpAddr::from([127, 0, 0, 1]), bound.port()),
            "loopback-only".to_string(),
            false,
        ),
    };
    let inbound: Arc<Mutex<Vec<InboundPeer>>> = Arc::new(Mutex::new(Vec::new()));
    let stop = tokio_util::sync::CancellationToken::new();
    let stop_task = stop.clone();
    let inbound_task = inbound.clone();
    let host = Arc::new(host);
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = stop_task.cancelled() => break,
                accepted = listener.accept() => {
                    let stream = match accepted {
                        Ok(s) => s,
                        Err(_) => continue,
                    };
                    let conn = match ThunderConnection::handshake(stream, &identity, None).await {
                        Ok(c) => c,
                        Err(_) => continue,
                    };
                    // Record handshake-level inbound identity. Trust
                    // requires a VALIDATED Pair request (pending queue).
                    if let Ok(mut guard) = inbound_task.lock() {
                        if !guard.iter().any(|p| p.peer_id == conn.peer_id) {
                            guard.push(InboundPeer {
                                peer_id: conn.peer_id.clone(),
                                name: conn.peer_name.clone(),
                                first_seen_epoch: epoch_now(),
                            });
                        }
                    }
                    let host = host.clone();
                    let stop_conn = stop_task.clone();
                    tokio::spawn(async move {
                        // Fresh trust snapshot per connection: explicit
                        // operator Trust/Reject applies without restart.
                        // The host-wide stop token closes live sessions:
                        // stop aborts generation and drops the connection.
                        let peers = PeerStore::load();
                        let _ = host.serve_with_shutdown(conn, &peers, stop_conn).await;
                    });
                }
            }
        }
    });
    Ok(ThunderHostRuntime {
        endpoint,
        tcp_port: bound.port(),
        iface_name,
        all_endpoints,
        lan_available,
        model_id,
        model_name: String::new(),
        backend_label: String::new(),
        stop_token: stop,
        registry,
        inbound,
    })
}

/// Send a real pairing request: open a TCP connection, handshake
/// (unauthenticated — trust is what pairing ESTABLISHES), send
/// `Pair { code + our identity }`, await the host's verdict. Accept is
/// a `Pair` echo with EMPTY code plus the host identity; anything else
/// is rejection. Returns the host identity for fingerprint confirmation.
pub async fn send_pairing_request(
    identity: &ThunderIdentity,
    addr: SocketAddr,
    code: &str,
) -> Result<AcceptedHost, ThunderError> {
    use crate::thunder::connection::{ThunderConnection, ThunderTransport};
    use crate::thunder::protocol::{MessageKind, ThunderMessage};
    let stream = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::net::TcpStream::connect(addr),
    )
    .await
    .map_err(|_| ThunderError::PeerUnavailable {
        peer: addr.to_string(),
    })?
    .map_err(|_| ThunderError::PeerUnavailable {
        peer: addr.to_string(),
    })?;
    let mut conn = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        ThunderConnection::handshake(stream, identity, None),
    )
    .await
    .map_err(|_| ThunderError::PeerUnavailable {
        peer: addr.to_string(),
    })??;
    let req = ThunderMessage::new(
        crate::thunder::receiver::next_request_id(),
        MessageKind::Pair {
            code: code.to_string(),
            peer_id: identity.peer_id.clone(),
            name: identity.display_name.clone(),
            public_key: identity.public_key_bytes().to_vec(),
        },
    );
    conn.send(&req).await?;
    let reply = tokio::time::timeout(std::time::Duration::from_secs(10), conn.receive())
        .await
        .map_err(|_| ThunderError::PeerUnavailable {
            peer: addr.to_string(),
        })??;
    let _ = conn.shutdown().await;
    match reply.payload {
        MessageKind::Pair {
            code,
            peer_id,
            name,
            public_key,
        } if code.is_empty() => {
            if peer_id.trim().is_empty() || public_key.len() != 32 {
                return Err(ThunderError::InvalidRequest {
                    detail: "pairing accept malformed: bad host identity".to_string(),
                });
            }
            let mut key = [0u8; 32];
            key.copy_from_slice(&public_key);
            Ok(AcceptedHost {
                peer_id,
                name,
                public_key: key,
            })
        }
        MessageKind::Error { code, message } => Err(ThunderError::InvalidRequest {
            detail: format!("pairing rejected by host ({code}): {message}"),
        }),
        _ => Err(ThunderError::InvalidRequest {
            detail: "unexpected reply to Pair request".to_string(),
        }),
    }
}

/// One-shot remote model fetch using the existing connection
/// primitives: handshake against the EXACT trusted record, send a
/// Models request, await the Models reply, close. No second reader,
/// no new protocol — the host's existing `serve` answers Models.
pub async fn fetch_remote_models(
    identity: &ThunderIdentity,
    trusted: &TrustedPeer,
    addr: SocketAddr,
) -> Result<Vec<ThunderModel>, ThunderError> {
    use crate::thunder::connection::{ThunderConnection, ThunderTransport};
    use crate::thunder::protocol::{MessageKind, ThunderMessage};
    let mut conn = ThunderConnection::connect(identity, trusted, addr).await?;
    let req = ThunderMessage::new(
        crate::thunder::receiver::next_request_id(),
        MessageKind::Models { models: Vec::new() },
    );
    conn.send(&req).await?;
    let reply = tokio::time::timeout(std::time::Duration::from_secs(10), conn.receive())
        .await
        .map_err(|_| ThunderError::PeerUnavailable {
            peer: trusted.peer_id.clone(),
        })??;
    let _ = conn.shutdown().await;
    match reply.payload {
        MessageKind::Models { models } => Ok(models),
        MessageKind::Error { code, message } => Err(ThunderError::InvalidRequest {
            detail: format!("models fetch: {code}: {message}"),
        }),
        _ => Err(ThunderError::InvalidRequest {
            detail: "unexpected reply to Models request".to_string(),
        }),
    }
}

/// Generate a fresh pairing secret and install it in the live
/// registry. Returns the displayable code text.
pub fn regenerate_pairing_code(runtime: &ThunderHostRuntime) -> String {
    let code = PairingCode::generate();
    let text = code.code.clone();
    runtime.registry.set_code(Some(code));
    text
}

/// One selectable Thunder action. Render and keyboard/mouse
/// activation share the same ordered list per tab.
#[derive(Debug, Clone)]
pub enum ThunderAction {
    GotoTab(usize),
    ToggleInference,
    CycleConcurrency,
    HostStart,
    HostStop,
    RegenCode,
    EditManual,
    SubmitManual,
    SelectConnectPeer(String),
    EditCode,
    SubmitCode,
    TrustHost,
    DiscardPending,
    TrustPending(String),
    RejectPending(String),
    DismissInbound(String),
    PeerToggleInference(String),
    PeerToggleForwarding(String),
    PeerUntrust(String),
    ModelsRefresh,
    SelectRemote(usize),
    SetMainAi,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_wire_ids() {
        assert_eq!(sanitize_wire_id("qwen3:32b"), Some("qwen3-32b".to_string()));
        assert_eq!(
            sanitize_wire_id("Qwen3VL-4B-Thinking-Q4_K_M.gguf"),
            Some("qwen3vl-4b-thinking-q4_k_m-gguf".to_string())
        );
        assert_eq!(sanitize_wire_id("!!!"), None);
        assert_eq!(sanitize_wire_id(""), None);
    }

    #[test]
    fn refuse_remote_backend_for_hosting() {
        let target = make_remote_target(
            "thunder-x",
            "127.0.0.1:1".parse().unwrap(),
            "m",
            TrustedPeer {
                peer_id: "thunder-x".to_string(),
                public_key: [0u8; 32],
                name: "X".to_string(),
                permissions: crate::thunder::pairing::PeerPermissions::default(),
                paired_at_epoch: 0,
            },
        );
        let backend = crate::backend::AgentBackend::SharedThunder(
            crate::thunder::runtime::SharedThunderBackend::new(
                Arc::new(ThunderIdentity::generate("t".to_string())),
                target,
            ),
        );
        assert!(host_model_from_backend(&backend).is_err());
    }

    #[test]
    fn ollama_backend_yields_real_identity() {
        let backend = crate::backend::AgentBackend::Ollama(crate::backend::OllamaBackend::new(
            "qwen3:8b".to_string(),
        ));
        let hosted = host_model_from_backend(&backend).expect("ollama hosts");
        assert_eq!(hosted.wire_id, "qwen3-8b");
        assert_eq!(hosted.display_name, "qwen3:8b");
        assert_eq!(hosted.format, "Ollama");
        assert!(hosted.thunder_model().validate().is_ok());
    }

    #[test]
    fn remote_ref_round_trips_peer_model_addr() {
        let addr: SocketAddr = "127.0.0.1:4317".parse().unwrap();
        let r = remote_model_ref("thunder-abc123", "qwen3-32b", addr);
        assert_eq!(r, "thunder-abc123::qwen3-32b@127.0.0.1:4317");
    }

    #[test]
    fn target_keeps_exact_peer_never_first() {
        let addr: SocketAddr = "127.0.0.1:4317".parse().unwrap();
        let alice = TrustedPeer {
            peer_id: "thunder-alice".to_string(),
            public_key: [1u8; 32],
            name: "Alice".to_string(),
            permissions: crate::thunder::pairing::PeerPermissions::default(),
            paired_at_epoch: 1,
        };
        let t = make_remote_target("thunder-alice", addr, "qwen3-32b", alice.clone());
        assert_eq!(t.peer_id, "thunder-alice");
        assert_eq!(t.address, addr);
        assert_eq!(t.model_id, "qwen3-32b");
        assert_eq!(t.trusted_peer.peer_id, "thunder-alice");
        assert_eq!(t.trusted_peer.public_key, [1u8; 32]);
    }

    #[test]
    fn tabs_cover_six_sections() {
        assert_eq!(ThunderTab::ALL.len(), 6);
        assert_eq!(ThunderTab::from_index(99), ThunderTab::Settings);
        assert_eq!(ThunderTab::Host.index(), 1);
    }

    #[test]
    fn health_labels_are_honest() {
        let mut h = PeerHealth::default();
        assert_eq!(h.label(false, true), "discovered·unverified");
        assert_eq!(h.label(false, false), "stale·unverified");
        assert_eq!(h.label(true, true), "trusted");
        h.record_ok();
        assert_eq!(h.label(true, true), "trusted·online");
        h.record_err("boom".to_string());
        assert_eq!(h.label(true, true), "trusted·unreachable");
    }

    #[test]
    fn iface_sort_prefers_private_lan_deterministically() {
        use std::net::{IpAddr, Ipv4Addr};
        let mut addrs = vec![
            IfaceAddr {
                name: "eth1".to_string(),
                ip: IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)),
            },
            IfaceAddr {
                name: "eth0".to_string(),
                ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 20)),
            },
            IfaceAddr {
                name: "vpn0".to_string(),
                ip: IpAddr::V4(Ipv4Addr::new(10, 8, 0, 2)),
            },
            IfaceAddr {
                name: "eth2".to_string(),
                ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 5)),
            },
        ];
        sort_iface_addrs(&mut addrs);
        // Private LAN first (10/8 before 192.168/16), then public;
        // same subnet sorts numerically; fully deterministic.
        let ips: Vec<String> = addrs.iter().map(|a| a.ip.to_string()).collect();
        assert_eq!(
            ips,
            vec!["10.8.0.2", "192.168.1.5", "192.168.1.20", "8.8.8.8"]
        );
        // Shuffled input converges to the same order.
        let mut shuffled = vec![
            IfaceAddr {
                name: "eth2".to_string(),
                ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 5)),
            },
            IfaceAddr {
                name: "vpn0".to_string(),
                ip: IpAddr::V4(Ipv4Addr::new(10, 8, 0, 2)),
            },
            IfaceAddr {
                name: "eth1".to_string(),
                ip: IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)),
            },
            IfaceAddr {
                name: "eth0".to_string(),
                ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 20)),
            },
        ];
        sort_iface_addrs(&mut shuffled);
        assert_eq!(
            shuffled
                .iter()
                .map(|a| a.ip.to_string())
                .collect::<Vec<_>>(),
            ips
        );
    }

    #[test]
    fn selected_address_is_never_loopback() {
        // On a real machine: either a usable address (never loopback /
        // unspecified) or None (caller shows loopback-only explicitly).
        if let Some(sel) = select_host_address() {
            assert!(!sel.ip.is_loopback(), "never select loopback as LAN");
            assert!(!sel.ip.is_unspecified(), "never select unspecified");
            assert!(!sel.name.is_empty(), "interface named");
        }
        for a in local_interface_addrs() {
            assert!(!a.ip.is_loopback(), "enumeration excludes loopback");
            assert!(!a.ip.is_unspecified(), "enumeration excludes unspecified");
        }
    }

    #[test]
    fn unknown_metadata_stays_unknown_never_fabricated() {
        let backend = crate::backend::AgentBackend::Ollama(crate::backend::OllamaBackend::new(
            "qwen3:8b".to_string(),
        ));
        let hosted = host_model_from_backend(&backend).expect("ollama hosts");
        // Architecture was never queried → explicit unknown, not a guess.
        assert_eq!(hosted.architecture, "unknown");
        let wire = hosted.thunder_model();
        assert_eq!(wire.architecture, "unknown");
        // Quantization genuinely unavailable → None, never invented.
        assert_eq!(wire.quantization, None);
    }

    #[test]
    fn manual_entry_carries_no_identity_for_trust() {
        // A manual endpoint has no peer id and no key: there is nothing
        // the Trust path can consume — `peer_key` marks it manual and
        // the pairing registry has no validated request for it.
        let addr: SocketAddr = "192.168.1.5:47832".parse().unwrap();
        assert_eq!(ThunderUiState::peer_key("", addr), format!("manual:{addr}"));
        let registry = crate::thunder::host::PairingRegistry::new();
        assert!(registry.consume(&format!("manual:{addr}")).is_none());
        assert!(registry.pending().is_empty());
    }
}
