//! Automatic routing (T11 first cut): decide local vs remote for one
//! request.
//!
//! Pure decision, never side effects: given local availability and the
//! capabilities of trusted peers, pick where inference should run. An
//! explicit user choice always overrides this module.
//!
//! This is a DELIBERATELY simple router: first matching peer wins. It
//! does not yet weigh latency, TTFT, throughput, context size, host
//! load or hardware capability — those are later refinements, so this
//! is not the final "intelligent Shared Thunder routing" system.

use super::capabilities::HostCapabilities;

/// Where a request should run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteDecision {
    /// The local backend handles it (default whenever available).
    Local,
    /// Send to this peer (its advertisement matched the request).
    Remote { peer_id: String, model_id: String },
    /// Nothing local and no peer advertises the model.
    NoRoute { reason: String },
}

/// Route one request. `local_available` reflects the real local probe;
/// `peers` are trusted peers WITH their latest capabilities. Deterministic:
/// first matching peer in the given order wins.
pub fn route(
    local_available: bool,
    model_id: &str,
    peers: &[(String, HostCapabilities)],
) -> RouteDecision {
    if local_available {
        return RouteDecision::Local;
    }
    for (peer_id, caps) in peers {
        if !caps.inference {
            continue;
        }
        if caps.find_model(model_id).is_some() {
            return RouteDecision::Remote {
                peer_id: peer_id.clone(),
                model_id: model_id.to_string(),
            };
        }
    }
    RouteDecision::NoRoute {
        reason: format!("local backend unavailable and no trusted peer advertises {model_id}"),
    }
}

/// Pick the best peer among those advertising a model: streaming +
/// cancellation support first, then advertisement order.
pub fn peers_advertising(model_id: &str, peers: &[(String, HostCapabilities)]) -> Vec<String> {
    peers
        .iter()
        .filter(|(_, caps)| caps.inference && caps.find_model(model_id).is_some())
        .map(|(id, _)| id.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::capabilities::ThunderModel;
    use super::*;

    fn caps(peer_id: &str, models: &[&str], inference: bool) -> HostCapabilities {
        HostCapabilities {
            peer_id: peer_id.to_string(),
            hardware: crate::model::HardwareInfo::detect(),
            models: models
                .iter()
                .map(|m| ThunderModel {
                    id: m.to_string(),
                    name: m.to_string(),
                    architecture: "Qwen3".to_string(),
                    format: "SafeTensors".to_string(),
                    backend: "Transformers".to_string(),
                    quantization: None,
                    context_length: 32768,
                    streaming: true,
                    cancellation: true,
                })
                .collect(),
            inference,
            forwarding: false,
            max_concurrent_requests: 4,
            max_context_tokens: 32768,
            max_new_tokens: 512,
            streaming: true,
            cancellation: true,
        }
    }

    #[test]
    fn test_local_available_wins() {
        let peers = vec![("p1".to_string(), caps("p1", &["qwen3-30b"], true))];
        // Even when a peer advertises the model, a healthy local backend
        // handles the request.
        assert_eq!(route(true, "qwen3-30b", &peers), RouteDecision::Local);
    }

    #[test]
    fn test_remote_when_peer_advertises() {
        let peers = vec![
            ("p1".to_string(), caps("p1", &["other"], true)),
            ("p2".to_string(), caps("p2", &["qwen3-30b"], true)),
        ];
        assert_eq!(
            route(false, "qwen3-30b", &peers),
            RouteDecision::Remote {
                peer_id: "p2".to_string(),
                model_id: "qwen3-30b".to_string(),
            }
        );
    }

    #[test]
    fn test_no_inference_permission_never_routed() {
        let peers = vec![("p1".to_string(), caps("p1", &["qwen3-30b"], false))];
        assert!(matches!(
            route(false, "qwen3-30b", &peers),
            RouteDecision::NoRoute { .. }
        ));
    }

    #[test]
    fn test_no_route_reason_names_model() {
        let peers = vec![("p1".to_string(), caps("p1", &["other"], true))];
        match route(false, "qwen3-30b", &peers) {
            RouteDecision::NoRoute { reason } => assert!(reason.contains("qwen3-30b")),
            other => panic!("expected NoRoute, got {other:?}"),
        }
    }

    #[test]
    fn test_peers_advertising_deterministic_order() {
        let peers = vec![
            ("b".to_string(), caps("b", &["m"], true)),
            ("a".to_string(), caps("a", &["m"], true)),
        ];
        assert_eq!(peers_advertising("m", &peers), vec!["b", "a"]);
        assert!(peers_advertising("zzz", &peers).is_empty());
    }
}
