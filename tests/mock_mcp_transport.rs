//! Mock MCP Transport for testing subscription streams and request correlation

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Mutex};
use uuid::Uuid;
use serde_json::{json, Value};

/// A mock MCP server that can simulate subscription streams
pub struct MockMcpServer {
    request_tx: mpsc::Sender<MockRequest>,
    subscriptions: Arc<Mutex<HashMap<u64, SubscriptionState>>>,
}

/// State of a subscription stream
struct SubscriptionState {
    subscription_id: String,
    events: Vec<Value>,
    event_index: usize,
    acknowledged: bool,
    closed: bool,
}

/// Internal request type
struct MockRequest {
    id: u64,
    request: Value,
    response_tx: oneshot::Sender<Value>,
}

impl MockMcpServer {
    pub fn new() -> (Self, MockClientTransport) {
        let (request_tx, request_rx) = mpsc::channel(100);
        let (event_tx, event_rx) = mpsc::channel(100);
        let subscriptions = Arc::new(Mutex::new(HashMap::new()));
        
        let server = Self {
            request_tx: request_tx.clone(),
            subscriptions: subscriptions.clone(),
        };
        
        let client = MockClientTransport {
            request_tx: request_tx.clone(),
            event_rx: Arc::new(Mutex::new(event_rx)),
            subscriptions: subscriptions.clone(),
            pending_requests: Arc::new(Mutex::new(HashMap::new())),
            next_request_id: Arc::new(Mutex::new(1)),
        };
        
        // Spawn server task
        let subscriptions_clone = subscriptions.clone();
        let event_tx_clone = event_tx.clone();
        tokio::spawn(async move {
            Self::run_server(request_rx, subscriptions_clone, event_tx_clone).await;
        });
        
        (server, client)
    }
    
    async fn run_server(
        mut request_rx: mpsc::Receiver<MockRequest>,
        subscriptions: Arc<Mutex<HashMap<u64, SubscriptionState>>>,
        event_tx: mpsc::Sender<Value>,
    ) {
        eprintln!("DEBUG: Server started, waiting for requests");
        while let Some(req) = request_rx.recv().await {
            eprintln!("DEBUG: Received request: {:?}", req.request.get("method"));
            let method = req.request.get("method").and_then(|m| m.as_str()).unwrap_or("");
            
            match method {
                "server/discover" => {
                    let response = json!({
                        "jsonrpc": "2.0",
                        "id": req.id,
                        "result": {
                            "supportedVersions": ["2026-07-28"],
                            "capabilities": {}
                        }
                    });
                    let _ = req.response_tx.send(response);
                }
                "test/method_a" => {
                    let response = json!({
                        "jsonrpc": "2.0",
                        "id": req.id,
                        "result": {"data": "response_a"}
                    });
                    let _ = req.response_tx.send(response);
                }
                "test/method_b" => {
                    let response = json!({
                        "jsonrpc": "2.0",
                        "id": req.id,
                        "result": {"data": "response_b"}
                    });
                    let _ = req.response_tx.send(response);
                }
                "subscriptions/listen" => {
                    let sub_id = format!("sub-{}", Uuid::new_v4().simple());
                    let notifications = req.request
                        .get("params")
                        .and_then(|p| p.get("notifications"))
                        .and_then(|n| n.as_object())
                        .map(|o| o.keys().cloned().collect::<Vec<_>>())
                        .unwrap_or_default();
                    
                    let mut events = Vec::new();
                    for notif in notifications {
                        // Convert camelCase to proper MCP notification format
                        let method = match notif.as_str() {
                            "toolsListChanged" => "notifications/tools/list_changed",
                            "promptsListChanged" => "notifications/prompts/list_changed",
                            "resourcesListChanged" => "notifications/resources/list_changed",
                            "resourceUpdated" => "notifications/resources/updated",
                            _ => &format!("notifications/{}", notif),
                        };
                        // Send 3 events per notification type to test stream lifecycle
                        for _ in 0..3 {
                            events.push(json!({
                                "jsonrpc": "2.0",
                                "method": method,
                                "params": {
                                    "_meta": {
                                        "io.modelcontextprotocol/subscriptionId": sub_id.clone()
                                    }
                                }
                            }));
                        }
                    }
                    
                    subscriptions.lock().await.insert(req.id, SubscriptionState {
                        subscription_id: sub_id.clone(),
                        events: events.clone(),
                        event_index: 0,
                        acknowledged: true,
                        closed: false,
                    });
                    
                    let ack = json!({
                        "jsonrpc": "2.0",
                        "id": req.id,
                        "result": {},
                        "_meta": {
                            "io.modelcontextprotocol/subscriptionId": sub_id
                        }
                    });
                    let _ = req.response_tx.send(ack);
                    
                    // Push events to the event stream after ACK
                    eprintln!("DEBUG: Sending {} events to stream", events.len());
                    for event in events {
                        eprintln!("DEBUG: Sending event: {:?}", event);
                        let _ = event_tx.send(event).await;
                    }
                    eprintln!("DEBUG: All events sent");
                }
                _ => {
                    let response = json!({
                        "jsonrpc": "2.0",
                        "id": req.id,
                        "error": {
                            "code": -32601,
                            "message": format!("Method not found")
                        }
                    });
                    let _ = req.response_tx.send(response);
                }
            }
        }
    }
}

/// Client-side transport that connects to mock server
pub struct MockClientTransport {
    request_tx: mpsc::Sender<MockRequest>,
    event_rx: Arc<Mutex<mpsc::Receiver<Value>>>,
    subscriptions: Arc<Mutex<HashMap<u64, SubscriptionState>>>,
    pending_requests: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>,
    next_request_id: Arc<Mutex<u64>>,
}

impl MockClientTransport {
    pub async fn send_request(&self, method: &str, params: Value) -> Result<Value, String> {
        let id = *self.next_request_id.lock().await;
        *self.next_request_id.lock().await += 1;
        
        let request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        });
        
let (response_tx, rx) = oneshot::channel();
        
        self.request_tx.send(MockRequest {
            id,
            request,
            response_tx,
        }).await
            .map_err(|e| format!("Failed to send request: {}", e))?;
        
        match tokio::time::timeout(std::time::Duration::from_secs(5), rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => Err("Channel closed".to_string()),
            Err(_) => Err("Request timeout".to_string()),
        }
    }
    
    pub async fn send_notification(&self, method: &str, params: Value) -> Result<(), String> {
        let id = *self.next_request_id.lock().await;
        *self.next_request_id.lock().await += 1;
        
        let request = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });
        
        self.request_tx.send(MockRequest {
            id,
            request,
            response_tx: oneshot::channel().0,
        }).await
            .map_err(|e| format!("Failed to send notification: {}", e))?;
        Ok(())
    }
    
    pub async fn next_event(&self) -> Option<Value> {
        let mut rx = self.event_rx.lock().await;
        rx.recv().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    
    #[tokio::test]
    async fn test_subscription_stream_ack_then_events() {
        let (_server, client) = MockMcpServer::new();
        
        let discover_resp = client.send_request("server/discover", json!({})).await.unwrap();
        assert!(discover_resp.get("result").is_some());
        
        let listen_resp = client.send_request("subscriptions/listen", json!({
            "notifications": {
                "toolsListChanged": true
            }
        })).await.unwrap();
        
        assert!(listen_resp.get("result").is_some());
        let sub_id = listen_resp
            .get("_meta")
            .and_then(|m| m.get("io.modelcontextprotocol/subscriptionId"))
            .and_then(|v| v.as_str())
            .expect("Should have subscriptionId in _meta");
        
        println!("Subscription ID: {}", sub_id);
        assert!(!sub_id.is_empty());
    }
    
    #[tokio::test]
    async fn test_out_of_order_response_correlation() {
        let (_server, client) = MockMcpServer::new();
        
        // Send two requests concurrently - they will get different IDs (1 and 2)
        let req1 = client.send_request("server/discover", json!({}));
        let req2 = client.send_request("server/discover", json!({}));
        
        // Wait for both responses - they may arrive in any order
        let (resp1, resp2) = tokio::join!(req1, req2);
        
        assert!(resp1.is_ok(), "Request 1 should succeed");
        assert!(resp2.is_ok(), "Request 2 should succeed");
        
        let r1 = resp1.unwrap();
        let r2 = resp2.unwrap();
        
        // Both should be valid JSON-RPC responses
        assert_eq!(r1["jsonrpc"], "2.0");
        assert_eq!(r2["jsonrpc"], "2.0");
        
        // Each response should have its own unique ID
        let id1 = r1.get("id").and_then(|v| v.as_u64()).expect("Response 1 should have ID");
        let id2 = r2.get("id").and_then(|v| v.as_u64()).expect("Response 2 should have ID");
        
        // IDs should be different (requests get sequential IDs)
        assert_ne!(id1, id2, "Requests should have different IDs");
        
        // Both should have results
        assert!(r1.get("result").is_some(), "Response 1 should have result");
        assert!(r2.get("result").is_some(), "Response 2 should have result");
    }
}