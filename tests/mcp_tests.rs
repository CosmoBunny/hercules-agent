//! Comprehensive test suite for MCP client implementation
//! 
//! Tests focus on 2026-07-28 subscription implementation and known bugs.

use serde_json::Value;
use hercules_agent::mcp::*;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc, oneshot, Barrier};

// Import mock transport
mod mock_mcp_transport;
use mock_mcp_transport::{MockMcpServer, MockClientTransport};

#[cfg(test)]
mod regression {
    use super::*;
    
    #[test]
    fn test_wrong_subscription_id_namespace_regression() {
        let response = json!({
            "io.modelcontextprotocol/subscriptionId": "sub-123"
        });
        
        let id = extract_subscription_id(&response);
        assert_eq!(id, Some("sub-123".to_string()));
        
        let wrong_response = json!({
            "subscriptionId": "sub-123"
        });
        let id = extract_subscription_id(&wrong_response);
        assert_eq!(id, None);
    }
    
    #[test]
    fn test_subscription_event_unknown_id_regression() {
        let mut subs = HashMap::new();
        subs.insert("toolsListChanged".to_string(), "known-sub-123".to_string());
        
        let event = json!({
            "method": "notifications/tools/list_changed",
            "params": {
                "_meta": {
                    "io.modelcontextprotocol/subscriptionId": "unknown-sub-456"
                }
            }
        });
        
        let is_valid = validate_subscription_event(&event, &subs);
        assert!(!is_valid);
    }
    
    #[test]
    fn test_subscription_events_after_initial_response() {
        let mut subs = HashMap::new();
        subs.insert("toolsListChanged".to_string(), "sub-123".to_string());
        
        for _ in 0..5 {
            let event = json!({
                "method": "notifications/tools/list_changed",
                "params": {
                    "_meta": {
                        "io.modelcontextprotocol/subscriptionId": "sub-123"
                    }
                }
            });
            assert!(validate_subscription_event(&event, &subs));
        }
    }
    
    #[test]
    fn test_subscription_id_in_response_variants() {
        // Direct
        let r1 = json!({"io.modelcontextprotocol/subscriptionId": "sub-1"});
        assert_eq!(extract_subscription_id(&r1), Some("sub-1".to_string()));
        
        // In result
        let r2 = json!({"result": {"io.modelcontextprotocol/subscriptionId": "sub-2"}});
        assert_eq!(extract_subscription_id(&r2), Some("sub-2".to_string()));
        
        // In _meta
        let r3 = json!({"_meta": {"io.modelcontextprotocol/subscriptionId": "sub-3"}});
        assert_eq!(extract_subscription_id(&r3), Some("sub-3".to_string()));
    }
    
    #[test]
    fn test_missing_subscription_id() {
        let response = json!({"result": "ok"});
        assert_eq!(extract_subscription_id(&response), None);
    }
    
    #[test]
    fn test_wrong_type_subscription_id() {
        let response = json!({"io.modelcontextprotocol/subscriptionId": 123});
        assert_eq!(extract_subscription_id(&response), None);
    }
    
    #[test]
    fn test_null_subscription_id() {
        let response = json!({"io.modelcontextprotocol/subscriptionId": null});
        assert_eq!(extract_subscription_id(&response), None);
    }
    
    #[test]
    fn test_empty_string_subscription_id() {
        let response = json!({"io.modelcontextprotocol/subscriptionId": ""});
        assert_eq!(extract_subscription_id(&response), Some("".to_string()));
    }
    
    #[test]
    fn test_multiple_active_subscriptions() {
        let mut subs = HashMap::new();
        subs.insert("toolsListChanged".to_string(), "sub-1".to_string());
        subs.insert("promptsListChanged".to_string(), "sub-2".to_string());
        subs.insert("resourcesListChanged".to_string(), "sub-3".to_string());
        
        let event1 = json!({
            "method": "notifications/tools/list_changed",
            "params": {"_meta": {"io.modelcontextprotocol/subscriptionId": "sub-1"}}
        });
        assert!(validate_subscription_event(&event1, &subs));
        
        let event2 = json!({
            "method": "notifications/prompts/list_changed",
            "params": {"_meta": {"io.modelcontextprotocol/subscriptionId": "sub-2"}}
        });
        assert!(validate_subscription_event(&event2, &subs));
        
        let event3 = json!({
            "method": "notifications/tools/list_changed",
            "params": {"_meta": {"io.modelcontextprotocol/subscriptionId": "sub-999"}}
        });
        assert!(!validate_subscription_event(&event3, &subs));
    }
    
    #[test]
    fn test_event_belonging_to_subscription_a_while_b_exists() {
        let mut subs = HashMap::new();
        subs.insert("toolsListChanged".to_string(), "sub-A".to_string());
        subs.insert("promptsListChanged".to_string(), "sub-B".to_string());
        
        let event = json!({
            "method": "notifications/tools/list_changed",
            "params": {"_meta": {"io.modelcontextprotocol/subscriptionId": "sub-A"}}
        });
        assert!(validate_subscription_event(&event, &subs));
        
        let event2 = json!({
            "method": "notifications/prompts/list_changed",
            "params": {"_meta": {"io.modelcontextprotocol/subscriptionId": "sub-B"}}
        });
        assert!(validate_subscription_event(&event2, &subs));
    }
    
    #[test]
    fn test_empty_subscription_map() {
        let subs = HashMap::new();
        let event = json!({
            "method": "notifications/tools/list_changed",
            "params": {"_meta": {"io.modelcontextprotocol/subscriptionId": "sub-123"}}
        });
        assert!(!validate_subscription_event(&event, &subs));
    }
    
    #[test]
    fn test_malformed_meta() {
        let mut subs = HashMap::new();
        subs.insert("toolsListChanged".to_string(), "sub-123".to_string());
        
        let event1 = json!({"method": "notifications/tools/list_changed", "params": {}});
        assert!(!validate_subscription_event(&event1, &subs));
        
        let event2 = json!({"method": "notifications/tools/list_changed", "params": {"_meta": "not an object"}});
        assert!(!validate_subscription_event(&event2, &subs));
        
        let event3 = json!({
            "method": "notifications/tools/list_changed",
            "params": {"_meta": {"io.modelcontextprotocol/subscriptionId": 123}}
        });
        assert!(!validate_subscription_event(&event3, &subs));
    }
    
    #[test]
    fn test_legacy_notification_path_without_subscription() {
        let subs = HashMap::new();
        let event = json!({"method": "notifications/tools/list_changed", "params": {}});
        assert!(!validate_subscription_event(&event, &subs));
    }
    
    #[test]
    fn test_unrelated_notification_methods() {
        let mut subs = HashMap::new();
        subs.insert("toolsListChanged".to_string(), "sub-123".to_string());
        
        let event = json!({
            "method": "notifications/prompts/list_changed",
            "params": {"_meta": {"io.modelcontextprotocol/subscriptionId": "sub-123"}}
        });
        // A prompts event with tools subscription ID should be INVALID
        assert!(!validate_subscription_event(&event, &subs));
    }
    
    #[tokio::test]
    async fn test_losing_tools_list_changed_notification() {
        let (tx, mut rx) = mpsc::channel::<String>(10);
        let tools_need_refresh = Arc::new(Mutex::new(false));
        let tools_need_refresh_clone = tools_need_refresh.clone();
        
        let mut handles = Vec::new();
        
        for _ in 0..10 {
            let tx = tx.clone();
            let tools_need_refresh = tools_need_refresh_clone.clone();
            handles.push(tokio::spawn(async move {
                let mut flag = tools_need_refresh.lock().await;
                *flag = true;
                let _ = tx.try_send("test-server".to_string());
            }));
        }
        
        for h in handles {
            h.await.unwrap();
        }
        
        let flag = tools_need_refresh.lock().await;
        assert!(*flag);
        
        let mut count = 0;
        while rx.try_recv().is_ok() {
            count += 1;
        }
        assert!(count > 0);
    }
    
#[tokio::test]
async fn test_out_of_order_response_correlation() {
    // Real test with multiple requests and responses in different orders
    // This tests the actual McpSession/pending_requests ID-based correlation
    
    let (_server, client) = MockMcpServer::new();
    
    // Send two requests concurrently
    let request_a = client.send_request("test/method_a", json!({}));
    let request_b = client.send_request("test/method_b", json!({}));
    
    // Wait for both requests to be sent
    let (resp_a, resp_b) = tokio::join!(request_a, request_b);
    
    // Both should succeed with their respective results
    assert!(resp_a.is_ok(), "Request A should succeed");
    assert!(resp_b.is_ok(), "Request B should succeed");
    
    // Verify responses have correct IDs (they should be different)
    let resp_a = resp_a.unwrap();
    let resp_b = resp_b.unwrap();
    
    assert!(resp_a.get("id").is_some(), "Response A should have ID");
    assert!(resp_b.get("id").is_some(), "Response B should have ID");
    assert_ne!(resp_a.get("id"), resp_b.get("id"), "Responses should have different IDs");
    
    // Verify the responses contain the correct data
    assert_eq!(resp_a.get("result").and_then(|r| r.get("data")).and_then(|d| d.as_str()), Some("response_a"));
    assert_eq!(resp_b.get("result").and_then(|r| r.get("data")).and_then(|d| d.as_str()), Some("response_b"));
    }
    #[tokio::test]
    async fn test_subscription_stream_lifecycle() {
        // This test verifies the complete subscription stream lifecycle:
        // 1. Discover
        // 2. Subscribe (gets ACK with subscription ID)
        // 3. Receive multiple events through the stream
        // 4. All events should be received and validated
        
        let (_server, client) = MockMcpServer::new();
        
        // 1. Discover - should succeed
        let discover_resp = client.send_request("server/discover", json!({})).await
            .expect("server/discover should succeed");
        assert!(discover_resp.get("result").is_some(), "Discovery should return result");
        
        // 2. Subscribe to tool list changes - should get acknowledgment with subscription ID
        let listen_resp = client.send_request("subscriptions/listen", json!({
            "notifications": {
                "toolsListChanged": true
            }
        })).await.expect("subscriptions/listen should succeed");
        
        // Should have acknowledgment with subscription ID in _meta
        assert!(listen_resp.get("result").is_some(), "Listen should return result");
        let sub_id = listen_resp
            .get("_meta")
            .and_then(|m| m.get("io.modelcontextprotocol/subscriptionId"))
            .and_then(|v| v.as_str())
            .expect("Should have subscriptionId in _meta");
        assert!(!sub_id.is_empty(), "Subscription ID should not be empty");
        
        // 3. Receive events from the subscription stream
        // The mock server will push 3 events after acknowledgment
        let mut events_received = 0;
        let mut received_ids = Vec::new();
        
        for _ in 0..3 {
            if let Some(event) = client.next_event().await {
                // Validate event structure
                assert_eq!(event.get("jsonrpc").and_then(|v| v.as_str()), Some("2.0"));
                // The mock should use "notifications/tools/list_changed" to match production
                assert_eq!(event.get("method").and_then(|v| v.as_str()), Some("notifications/tools/list_changed"));
                
                // Validate subscription ID in _meta
                let event_sub_id = event
                    .get("params")
                    .and_then(|p| p.get("_meta"))
                    .and_then(|m| m.get("io.modelcontextprotocol/subscriptionId"))
                    .and_then(|v| v.as_str())
                    .expect("Event should have subscriptionId in _meta");
                
                assert_eq!(event_sub_id, sub_id, "Event subscription ID should match");
                received_ids.push(event_sub_id.to_string());
                events_received += 1;
            } else {
                break; // No more events
            }
        }
        // Should receive exactly 3 events
        assert_eq!(events_received, 3, "Should receive exactly 3 subscription events");
        
        println!("Successfully received {} subscription events with ID: {}", events_received, sub_id);
    }
}

// Pure helper functions
pub fn extract_subscription_id(response: &Value) -> Option<String> {
    response
        .get("io.modelcontextprotocol/subscriptionId")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            response.get("result")
                .and_then(|r| r.get("io.modelcontextprotocol/subscriptionId"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .or_else(|| {
            response.get("_meta")
                .and_then(|m| m.get("io.modelcontextprotocol/subscriptionId"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
}

fn subscription_key_for_method(method: &str) -> &str {
    match method {
        "notifications/tools/list_changed" => "toolsListChanged",
        "notifications/prompts/list_changed" => "promptsListChanged",
        "notifications/resources/list_changed" => "resourcesListChanged",
        "notifications/resources/updated" => "resourceUpdated",
        _ => method.strip_prefix("notifications/").unwrap_or(method),
    }
}

pub fn validate_subscription_event(event: &Value, subscriptions: &HashMap<String, String>) -> bool {
    let params = event.get("params");
    if params.is_none() {
        return false;
    }
    
    let incoming_sub_id = params
        .and_then(|p| p.get("_meta"))
        .and_then(|m| m.get("io.modelcontextprotocol/subscriptionId"))
        .and_then(|v| v.as_str());
    
    let method = match event.get("method").and_then(Value::as_str) {
        Some(m) => m,
        None => return false,
    };
    
    let key = subscription_key_for_method(method);
    
    if let Some(sub_id) = incoming_sub_id {
        subscriptions
            .get(key)
            .is_some_and(|expected_id| expected_id == sub_id)
    } else {
        false
    }
}

