//! Model Context Protocol (MCP) Client Engine
//!
//! Provides a full asynchronous JSON-RPC 2.0 stdio client that manages MCP servers,
//! lifecycle handshakes (`initialize`, `notifications/initialized`), discovery (`tools/list`),
//! and tool execution (`tools/call`).

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{oneshot, Mutex, mpsc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

static NEXT_REQ_ID: AtomicU64 = AtomicU64::new(1);

/// Maximum size of a single MCP message (4 MB)
const MAX_MCP_MESSAGE_SIZE: usize = 4 * 1024 * 1024;

/// Supported MCP protocol versions (newest first for negotiation)
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &[
    "2026-07-28",
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
];

fn subscription_key_for_method(method: &str) -> &str {
    match method {
        "notifications/tools/list_changed" => "toolsListChanged",
        "notifications/prompts/list_changed" => "promptsListChanged",
        "notifications/resources/list_changed" => "resourcesListChanged",
        "notifications/resources/updated" => "resourceUpdated",
        _ => method.strip_prefix("notifications/").unwrap_or(method),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolCallResult {
    pub is_error: bool,
    pub content: Vec<McpContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { data: String, mime_type: String },
    #[serde(rename = "resource")]
    Resource { resource: Value },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone)]
pub enum McpToolCallOutcome {
    Complete {
        is_error: bool,
        content: Vec<McpContent>,
    },
    InputRequired {
        input_requests: Value,
        request_state: String,
    },
}

impl McpToolCallOutcome {
    pub fn to_plain_text(&self) -> String {
        match self {
            McpToolCallOutcome::Complete { is_error, content } => {
                let mut out = String::new();
                for item in content {
                    match item {
                        McpContent::Text { text } => {
                            if !out.is_empty() {
                                out.push('\n');
                            }
                            out.push_str(text);
                        }
                        McpContent::Image { mime_type, .. } => {
                            if !out.is_empty() {
                                out.push('\n');
                            }
                            out.push_str(&format!("[Binary image data ({mime_type})]"));
                        }
                        McpContent::Resource { resource } => {
                            if !out.is_empty() {
                                out.push('\n');
                            }
                            out.push_str(&format!("[Resource: {}]", resource));
                        }
                        McpContent::Unknown => {
                            if !out.is_empty() {
                                out.push('\n');
                            }
                            out.push_str("[Unknown content]");
                        }
                    }
                }
                if out.is_empty() {
                    if *is_error {
                        "Error: (empty error response from MCP tool)".to_string()
                    } else {
                        "Success (no output)".to_string()
                    }
                } else if *is_error {
                    format!("Error: {out}")
                } else {
                    out
                }
            }
            McpToolCallOutcome::InputRequired { .. } => {
                "[Input required - elicitation needed]".to_string()
            }
        }
    }
}

impl McpToolCallResult {
    pub fn to_plain_text(&self) -> String {
        let mut out = String::new();
        for item in &self.content {
            match item {
                McpContent::Text { text } => {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(text);
                }
                McpContent::Image { mime_type, .. } => {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(&format!("[Binary image data ({mime_type})]"));
                }
                McpContent::Resource { resource } => {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(&format!("[Resource: {}]", resource));
                }
                McpContent::Unknown => {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str("[Unknown content]");
                }
            }
        }
        if out.is_empty() {
            if self.is_error {
                "Error: (empty error response from MCP tool)".to_string()
            } else {
                "Success (no output)".to_string()
            }
        } else if self.is_error {
            format!("Error: {out}")
        } else {
            out
        }
    }
}

/// Result of protocol version negotiation
#[derive(Debug, Clone)]
struct ProtocolNegotiation {
    version: String,
    server_capabilities: Value,
    server_info: Value,
}

/// Handlers for server-initiated messages
type NotificationHandler = Arc<dyn Fn(Value) + Send + Sync>;
type RequestHandler = Arc<dyn Fn(u64, String, Value, mpsc::Sender<WriterCommand>) -> Result<(), String> + Send + Sync>;

/// Writer command for stdin writer task
#[derive(Debug)]
enum WriterCommand {
    Write(Value),
    Shutdown,
}

/// An active stdio session with an external MCP server
pub struct McpSession {
    pub server_name: String,
    pub tools: Vec<McpToolDefinition>,
    pub protocol_version: String,
    pub server_capabilities: Value,
    pub server_info: Value,
    stdin: Arc<Mutex<ChildStdin>>,
    writer_tx: mpsc::Sender<WriterCommand>,
    pending_requests: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, String>>>>>,
    child: Arc<Mutex<Child>>,
    notification_handlers: Arc<Mutex<HashMap<String, NotificationHandler>>>,
    request_handlers: Arc<Mutex<HashMap<String, RequestHandler>>>,
    tools_need_refresh: Arc<Mutex<bool>>,
    /// Active 2026 subscriptions: method -> subscription ID
    subscriptions: Arc<Mutex<HashMap<String, String>>>,
}

impl McpSession {
    /// Spawn an MCP server process, establish JSON-RPC 2.0 framing, perform initialize handshake, and list tools
    pub async fn spawn(
        server_name: String,
        command: &str,
        args: &[String],
        env_vars: Option<HashMap<String, String>>,
        tools_changed_tx: Option<mpsc::Sender<String>>,
    ) -> Result<Self, String> {
        let cmd_display = if args.is_empty() {
            command.to_string()
        } else {
            format!("{} {}", command, args.join(" "))
        };
        let mut command = Command::new(command);
        command.args(args);
        command.stdin(Stdio::piped());
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());

        if let Some(envs) = env_vars {
            for (k, v) in envs {
                command.env(k, v);
            }
        }

        let mut child = command.spawn().map_err(|e| {
            format!("Failed to spawn MCP server '{server_name}' (command: {cmd_display}): {e}")
        })?;

        let stdin = child.stdin.take().ok_or_else(|| "Failed to capture stdin of MCP process".to_string())?;
        let stdout = child.stdout.take().ok_or_else(|| "Failed to capture stdout of MCP process".to_string())?;
        let stderr = child.stderr.take();

        let stdin_arc = Arc::new(Mutex::new(stdin));
        let pending = Arc::new(Mutex::new(HashMap::<u64, oneshot::Sender<Result<Value, String>>>::new()));
        let notification_handlers = Arc::new(Mutex::new(HashMap::<String, NotificationHandler>::new()));
        let request_handlers = Arc::new(Mutex::new(HashMap::<String, RequestHandler>::new()));
        let subscriptions = Arc::new(Mutex::new(HashMap::<String, String>::new()));

        // Create writer channel and spawn writer task
        let (writer_tx, mut writer_rx) = mpsc::channel::<WriterCommand>(32);
        let stdin_writer = stdin_arc.clone();
        let s_name_writer = server_name.clone();
        tokio::spawn(async move {
            while let Some(cmd) = writer_rx.recv().await {
                match cmd {
                    WriterCommand::Write(val) => {
                        let mut line = match serde_json::to_string(&val) {
                            Ok(l) => l,
                            Err(e) => {
                                eprintln!("[MCP:{s_name_writer}:writer] JSON serialization error: {e}");
                                continue;
                            }
                        };
                        line.push('\n');
                        let mut stdin = stdin_writer.lock().await;
                        if let Err(e) = stdin.write_all(line.as_bytes()).await {
                            eprintln!("[MCP:{s_name_writer}:writer] Write error: {e}");
                            break;
                        }
                        if let Err(e) = stdin.flush().await {
                            eprintln!("[MCP:{s_name_writer}:writer] Flush error: {e}");
                            break;
                        }
                    }
                    WriterCommand::Shutdown => {
                        break;
                    }
                }
            }
        });

        // Background stderr logger
        if let Some(err_stream) = stderr {
            let s_name = server_name.clone();
            tokio::spawn(async move {
                let mut reader = BufReader::new(err_stream).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    eprintln!("[MCP:{s_name}:stderr] {line}");
                }
            });
        }

        // Background stdout reader for JSON-RPC 2.0 lines
        let pending_clone = Arc::clone(&pending);
        let notification_handlers_clone = Arc::clone(&notification_handlers);
        let request_handlers_clone = Arc::clone(&request_handlers);
        let subscriptions_clone = Arc::clone(&subscriptions);
        let tools_need_refresh = Arc::new(Mutex::new(false));
        let tools_need_refresh_clone = Arc::clone(&tools_need_refresh);
        let tools_changed_tx_clone = tools_changed_tx.clone();
        let writer_tx_clone = writer_tx.clone();
        let s_name_clone = server_name.clone();
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed.len() > MAX_MCP_MESSAGE_SIZE {
                    eprintln!("[MCP:{s_name_clone}:error] Message exceeds maximum size ({} bytes)", MAX_MCP_MESSAGE_SIZE);
                    continue;
                }
                if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
                    if let Some(id_val) = val.get("id") {
                        if let Some(id) = id_val.as_u64() {
                            if let Some(method) = val.get("method").and_then(|m| m.as_str()) {
                                // Server-initiated request with ID
                                let params = val.get("params").cloned();
                                let handlers = request_handlers_clone.lock().await;
                                if let Some(handler) = handlers.get(method) {
                                    // Pass writer channel to handler so it can respond
                                    let writer_tx = writer_tx_clone.clone();
                                    if let Err(e) = handler(id, method.to_string(), params.unwrap_or(Value::Null), writer_tx) {
                                        eprintln!("[MCP:{s_name_clone}:server_request] Handler error: {e}");
                                    }
                                } else {
                                    // No handler - send error response
                                    let error_response = json!({
                                        "jsonrpc": "2.0",
                                        "id": id,
                                        "error": {
                                            "code": -32601,
                                            "message": format!("Method not found: {method}")
                                        }
                                    });
                                    let _ = writer_tx_clone.send(WriterCommand::Write(error_response)).await;
                                    eprintln!("[MCP:{s_name_clone}:server_request] No handler for method: {method}");
                                }
                            } else {
                                // Response to our request (has id but no method)
                                let mut map = pending_clone.lock().await;
                                if let Some(tx) = map.remove(&id) {
                                    if let Some(err_obj) = val.get("error") {
                                        let msg = err_obj.get("message").and_then(|m| m.as_str()).unwrap_or("Unknown JSON-RPC error");
                                        let _ = tx.send(Err(msg.to_string()));
                                    } else if let Some(res) = val.get("result") {
                                        let _ = tx.send(Ok(res.clone()));
                                    } else {
                                        let _ = tx.send(Ok(Value::Null));
                                    }
                                }
                            }
                        }
                    } else if let Some(method) = val.get("method").and_then(|m| m.as_str()) {
                        // Notification (no id)
                        let params = val.get("params").cloned();
                        
                        // Check if this is a subscription event (2026 protocol)
                        // Subscription events carry subscription ID in _meta
                        let is_subscription_event = if let Some(p) = &params {
                            p.get("_meta")
                                .and_then(|m| m.get("io.modelcontextprotocol/subscriptionId"))
                                .is_some()
                        } else {
                            false
                        };
                        
                        if is_subscription_event {
                            // Check if we have active subscriptions (2026 protocol)
                            let has_subscriptions = {
                                let subs = subscriptions_clone.lock().await;
                                !subs.is_empty()
                            };
                            
                            if has_subscriptions {
                                // Extract subscriptionId from _meta (2026 namespaced key)
                                let incoming_sub_id = params.as_ref()
                                    .and_then(|p| p.get("_meta"))
                                    .and_then(|m| m.get("io.modelcontextprotocol/subscriptionId"))
                                    .and_then(|v| v.as_str());
                                
                                // Validate subscriptionId against stored subscriptions with method matching
                                let is_valid_subscription = if let Some(sub_id) = incoming_sub_id {
                                    let subs = subscriptions_clone.lock().await;
                                    let key = subscription_key_for_method(method);
                                    subs.get(key).is_some_and(|stored_id| stored_id == sub_id)
                                } else {
                                    false
                                };
                                
                                if is_valid_subscription {
                                    // Dispatch based on notification type
                                    match method {
                                        "notifications/tools/list_changed" => {
                                            eprintln!("[MCP:{s_name_clone}] Received toolsListChanged subscription event");
                                            let tools_need_refresh = tools_need_refresh_clone.clone();
                                            tokio::spawn(async move {
                                                let mut flag = tools_need_refresh.lock().await;
                                                *flag = true;
                                            });
                                            if let Some(ref tx) = tools_changed_tx_clone {
                                                let _ = tx.try_send(s_name_clone.clone());
                                            }
                                        }
                                        "notifications/prompts/list_changed" => {
                                            eprintln!("[MCP:{s_name_clone}] Received promptsListChanged subscription event");
                                            // Could trigger prompt refresh if needed
                                        }
                                        "notifications/resources/list_changed" => {
                                            eprintln!("[MCP:{s_name_clone}] Received resourcesListChanged subscription event");
                                            // Could trigger resource refresh if needed
                                        }
                                        "notifications/resources/updated" => {
                                            eprintln!("[MCP:{s_name_clone}] Received resourceUpdated subscription event");
                                            // Could trigger resource refresh if needed
                                        }
                                        _ => {
                                            eprintln!("[MCP:{s_name_clone}] Received unknown subscription event: {method}");
                                        }
                                    }
                                } else {
                                    eprintln!("[MCP:{s_name_clone}] Subscription event with unknown/invalid subscriptionId: {method}");
                                }
                            } else {
                                // No active subscriptions, treat as regular notification
                                let handlers = notification_handlers_clone.lock().await;
                                if let Some(handler) = handlers.get(method) {
                                    handler(params.unwrap_or(Value::Null));
                                } else {
                                    eprintln!("[MCP:{s_name_clone}:notification] Unhandled notification: {method}");
                                }
                            }
                        } else {
                            // Regular notification
                            let handlers = notification_handlers_clone.lock().await;
                            if let Some(handler) = handlers.get(method) {
                                handler(params.unwrap_or(Value::Null));
                            } else {
                                eprintln!("[MCP:{s_name_clone}:notification] Unhandled notification: {method}");
                            }
                        }
                    } else {
                        eprintln!("[MCP:{s_name_clone}:raw] {trimmed}");
                    }
                } else {
                    eprintln!("[MCP:{s_name_clone}:raw] {trimmed}");
                }
            }
        });

        let mut session = Self {
            server_name: server_name.clone(),
            tools: Vec::new(),
            protocol_version: String::new(),
            server_capabilities: Value::Null,
            server_info: Value::Null,
            stdin: stdin_arc,
            writer_tx,
            pending_requests: pending,
            child: Arc::new(Mutex::new(child)),
            notification_handlers,
            request_handlers,
            tools_need_refresh,
            subscriptions,
        };
        {
            let session_name = session.server_name.clone();
            let tools_changed_tx_clone = tools_changed_tx.clone();
            let tools_need_refresh_clone = session.tools_need_refresh.clone();
            let mut handlers = session.notification_handlers.lock().await;
            handlers.insert("notifications/tools/list_changed".to_string(), Arc::new(move |_params| {
                eprintln!("[MCP:{}] Received tools/list_changed notification", session_name);
                // Set the flag to trigger a refresh
                let tools_need_refresh = tools_need_refresh_clone.clone();
                tokio::spawn(async move {
                    let mut flag = tools_need_refresh.lock().await;
                    *flag = true;
                });
                if let Some(ref tx) = tools_changed_tx_clone {
                    // Try to send, ignore if channel is full or closed
                    let _ = tx.try_send(session_name.clone());
                }
            }));

            // NOTE: subscriptions/listen handler is registered AFTER protocol negotiation
            // below, since we need to know the protocol version first.
        }

        // Perform protocol negotiation
        let negotiation = session.negotiate_protocol().await
            .map_err(|e| format!("MCP '{server_name}' protocol negotiation failed: {e}"))?;

        session.protocol_version = negotiation.version;
        session.server_capabilities = negotiation.server_capabilities;
        session.server_info = negotiation.server_info;

        // Send `notifications/initialized` notification (no response expected)
        // Only for legacy protocol versions that require it
        if session.protocol_version != "2026-07-28" {
            session.send_notification("notifications/initialized", json!({})).await
                .map_err(|e| format!("MCP '{server_name}' initialized notification failed: {e}"))?;
        }

        // 2026-07-28: subscribe to change notifications via subscriptions/listen
        if session.protocol_version == "2026-07-28" {
            // Send subscriptions/listen request to open the change notification stream
            let listen_params = json!({
                "notifications": {
                    "toolsListChanged": true
                }
            });
            match session.send_request_with_meta("subscriptions/listen", listen_params, None).await {
                Ok(res) => {
                    // Extract subscription ID from response (2026 uses namespaced key)
                    let sub_id = res
                        .get("io.modelcontextprotocol/subscriptionId")
                        .and_then(|v| v.as_str())
                        .or_else(|| res.get("result").and_then(|r| r.get("io.modelcontextprotocol/subscriptionId")).and_then(|v| v.as_str()))
                        .or_else(|| res.get("_meta").and_then(|m| m.get("io.modelcontextprotocol/subscriptionId")).and_then(|v| v.as_str()));
                    if let Some(sub_id) = sub_id {
                        let mut subs = session.subscriptions.lock().await;
                        subs.insert("toolsListChanged".to_string(), sub_id.to_string());
                        eprintln!("[MCP:{}] Subscribed to toolsListChanged with ID: {}", session.server_name, sub_id);
                    } else {
                        eprintln!("[MCP:{}] subscriptions/listen succeeded but no io.modelcontextprotocol/subscriptionId in response: {}", session.server_name, res);
                    }
                }
                Err(e) => {
                    eprintln!("[MCP:{}] subscriptions/listen failed: {e}", session.server_name);
                }
            }
        }

        // Fetch available tools
        session.refresh_tools().await
            .map_err(|e| format!("MCP '{server_name}' tools/list failed: {e}"))?;

        Ok(session)
    }

    /// Negotiate MCP protocol version with the server
    async fn negotiate_protocol(&mut self) -> Result<ProtocolNegotiation, String> {
        // First try server/discover for 2026-07-28 (stateless protocol)
        // This is the modern way to check for 2026 support
        if SUPPORTED_PROTOCOL_VERSIONS.contains(&"2026-07-28") {
            eprintln!("[MCP:{}:info] Attempting server/discover for 2026-07-28", self.server_name);
            
            // Try server/discover to check for 2026 support
            match self.send_request_with_meta("server/discover", json!({}), None).await {
                Ok(res) => {
                    // Check if server supports 2026-07-28
                    if let Some(versions) = res.get("supportedVersions").and_then(|v| v.as_array()) {
                        let supports_2026 = versions.iter().any(|v| v.as_str() == Some("2026-07-28"));
                        if supports_2026 {
                            eprintln!("[MCP:{}:info] Server supports 2026-07-28", self.server_name);
                            
                            // For 2026, we don't need initialize handshake
                            // Just use the discovered version
                            let server_capabilities = res.get("capabilities").cloned().unwrap_or(Value::Null);
                            let server_info = res
                    .get("_meta")
                    .and_then(|m| m.get("io.modelcontextprotocol/serverInfo"))
                    .cloned()
                    .unwrap_or_else(|| res.get("serverInfo").cloned().unwrap_or(Value::Null));
                            
                            return Ok(ProtocolNegotiation {
                                version: "2026-07-28".to_string(),
                                server_capabilities,
                                server_info,
                            });
                        }
                    }
                }
                Err(e) => {
                    // server/discover failed, fall back to legacy initialize
                    eprintln!("[MCP:{}:info] server/discover failed: {e}, trying legacy initialize", self.server_name);
                }
            }
        }
        
        // Fall back to legacy initialize handshake for older versions
        for version in SUPPORTED_PROTOCOL_VERSIONS {
            if *version == "2026-07-28" {
                continue; // Already tried above
            }
            
            let init_params = json!({
                "protocolVersion": version,
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "hercules-agent",
                    "version": env!("CARGO_PKG_VERSION")
                }
            });

            match self.send_request("initialize", init_params).await {
                Ok(res) => {
                    let server_version = res.get("protocolVersion")
                        .and_then(|v| v.as_str())
                        .ok_or("Server did not return protocolVersion")?;
                    
                    // Verify the server returned a version we support
                    if !SUPPORTED_PROTOCOL_VERSIONS.contains(&server_version) {
                        return Err(format!("Server returned unsupported protocol version: {server_version}"));
                    }

                    let server_capabilities = res.get("capabilities").cloned().unwrap_or(Value::Null);
                    let server_info = res
                    .get("_meta")
                    .and_then(|m| m.get("io.modelcontextprotocol/serverInfo"))
                    .cloned()
                    .unwrap_or_else(|| res.get("serverInfo").cloned().unwrap_or(Value::Null));

                    return Ok(ProtocolNegotiation {
                        version: server_version.to_string(),
                        server_capabilities,
                        server_info,
                    });
                }
                Err(e) => {
                    // Try next version
                    eprintln!("[MCP:{}:warn] Protocol version {version} failed: {e}, trying next", self.server_name);
                    continue;
                }
            }
        }

        Err("Failed to negotiate any supported MCP protocol version".to_string())
    }

/// Send a JSON-RPC request and await matching response ID
    pub async fn send_request(&self, method: &str, params: Value) -> Result<Value, String> {
        let req_id = NEXT_REQ_ID.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();

        {
            let mut map = self.pending_requests.lock().await;
            map.insert(req_id, tx);
        }

        let mut body = json!({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params
        });

        // Add modern _meta envelope inside params only for 2026 protocol
        if self.protocol_version == "2026-07-28" {
            if let Some(params_map) = body["params"].as_object_mut() {
                params_map.insert(
                    "_meta".to_string(),
                    json!({
                        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                        "io.modelcontextprotocol/clientInfo": {
                            "name": "hercules-agent",
                            "version": env!("CARGO_PKG_VERSION")
                        },
                        "io.modelcontextprotocol/clientCapabilities": {}
                    })
                );
            }
        }

        // Use writer channel instead of direct stdin write
        self.writer_tx.send(WriterCommand::Write(body)).await
            .map_err(|e| format!("Failed to send to writer: {e}"))?;

        match tokio::time::timeout(std::time::Duration::from_secs(30), rx).await {
            Ok(Ok(res)) => res,
            Ok(Err(_)) => Err("MCP server closed connection without response".to_string()),
            Err(_) => {
                let mut map = self.pending_requests.lock().await;
                map.remove(&req_id);
                Err("MCP request timed out after 30s".to_string())
            }
        }
}
    
    /// Check if tools need refresh and refresh them
    pub async fn check_and_refresh_tools(&mut self) -> Result<bool, String> {
        let need_refresh = {
            let mut flag = self.tools_need_refresh.lock().await;
            if *flag {
                *flag = false;
                true
            } else {
                false
            }
        };
        
        if need_refresh {
            eprintln!("[MCP:{}] Refreshing tools due to list_changed notification", self.server_name);
            self.refresh_tools().await?;
            return Ok(true);
        }
        Ok(false)
    }
    
    /// Send a JSON-RPC request with optional _meta (for 2026 protocol)
    async fn send_request_with_meta(&self, method: &str, params: Value, meta: Option<Value>) -> Result<Value, String> {
        let req_id = NEXT_REQ_ID.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();

        {
            let mut map = self.pending_requests.lock().await;
            map.insert(req_id, tx);
        }

        let mut body = json!({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params
        });

        // Add modern _meta envelope inside params
        // Add for:
        //   - 2026 protocol (after negotiation)
        //   - when meta is explicitly provided (caller wants custom _meta)
        //   - for discovery probe (server/discover before negotiation)
        let is_discovery = method == "server/discover";
        let add_meta = self.protocol_version == "2026-07-28" || meta.is_some() || is_discovery;
        if add_meta && body["params"].is_object() {
            let mut meta_obj = json!({
                "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientInfo": {
                    "name": "hercules-agent",
                    "version": env!("CARGO_PKG_VERSION")
                },
                "io.modelcontextprotocol/clientCapabilities": {}
            });
            // Merge provided meta if present
            if let Some(m) = meta {
                if let Some(obj) = m.as_object() {
                    for (k, v) in obj {
                        meta_obj[k] = v.clone();
                    }
                }
            }
            body["params"]["_meta"] = meta_obj;
        }

        // Use writer channel instead of direct stdin write
        self.writer_tx.send(WriterCommand::Write(body)).await
            .map_err(|e| format!("Failed to send to writer: {e}"))?;

        match tokio::time::timeout(std::time::Duration::from_secs(30), rx).await {
            Ok(Ok(res)) => res,
            Ok(Err(_)) => Err("MCP server closed connection without response".to_string()),
            Err(_) => {
                let mut map = self.pending_requests.lock().await;
                map.remove(&req_id);
                Err("MCP request timed out after 30s".to_string())
            }
        }
    }
    
    /// Send a JSON-RPC notification (no id)
    pub async fn send_notification(&self, method: &str, params: Value) -> Result<(), String> {
        let mut body = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });

        // Add modern _meta envelope inside params only for 2026 protocol
        if self.protocol_version == "2026-07-28" {
            if body["params"].is_object() {
                body["params"]["_meta"] = json!({
                    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                    "io.modelcontextprotocol/clientInfo": {
                        "name": "hercules-agent",
                        "version": env!("CARGO_PKG_VERSION")
                    },
                    "io.modelcontextprotocol/clientCapabilities": {}
                });
            }
        }

        // Use writer channel instead of direct stdin write
        self.writer_tx.send(WriterCommand::Write(body)).await
            .map_err(|e| format!("Failed to send to writer: {e}"))?;
        Ok(())
    }

    /// Query `tools/list` and update local tool definitions (with pagination support)
    pub async fn refresh_tools(&mut self) -> Result<(), String> {
        let mut tools_out = Vec::new();
        let mut cursor: Option<String> = None;

        loop {
            let params = if let Some(ref c) = cursor {
                json!({ "cursor": c })
            } else {
                json!({})
            };

            let res = self.send_request("tools/list", params).await?;

            if let Some(tools_arr) = res.get("tools").and_then(|t| t.as_array()) {
                for t in tools_arr {
                    if let Some(name) = t.get("name").and_then(|n| n.as_str()) {
                        let desc = t.get("description").and_then(|d| d.as_str()).map(|s| s.to_string());
                        let schema = t.get("inputSchema").cloned().unwrap_or_else(|| json!({ "type": "object" }));
                        tools_out.push(McpToolDefinition {
                            name: name.to_string(),
                            description: desc,
                            input_schema: schema,
                        });
                    }
                }
            }

            // Check for next cursor
            cursor = res.get("nextCursor")
                .and_then(|c| c.as_str())
                .map(|s| s.to_string());

            if cursor.is_none() {
                break;
            }
        }

        self.tools = tools_out;
        Ok(())
    }

    /// Call an MCP tool on this server
    ///
    /// For 2026-07-28 protocol, if the tool returns `input_required`, the caller
    /// should provide `inputResponses` and `requestState` and retry the call.
    pub async fn call_tool(
        &self,
        tool_name: &str,
        arguments: Value,
        input_responses: Option<Value>,
        request_state: Option<String>,
    ) -> Result<McpToolCallOutcome, String> {
        // Validate arguments against tool schema
        if let Some(tool_def) = self.tools.iter().find(|t| t.name == tool_name) {
            if let Err(e) = Self::validate_basic_arguments(&tool_def.input_schema, &arguments) {
                return Err(format!("Argument validation failed for tool '{tool_name}': {e}"));
            }
        }

        let mut params = json!({
            "name": tool_name,
            "arguments": arguments
        });

        // Add inputResponses and requestState for MRTR retry if provided
        if let Some(ir) = input_responses {
            params["inputResponses"] = ir;
        }
        if let Some(rs) = request_state {
            params["requestState"] = json!(rs);
        }

        let res = self.send_request("tools/call", params).await?;
        
        // Handle 2026 MRTR: input_required resultType
        let result_type = res.get("resultType").and_then(|r| r.as_str()).unwrap_or("complete");

        let is_error = res.get("isError").and_then(|b| b.as_bool()).unwrap_or(false);

        if result_type == "input_required" {
            // Elicitation/sampling needed - return input_required outcome
            // The caller will need to provide more inputResponses and retry
            let input_requests = res.get("inputRequests").cloned().unwrap_or(Value::Null);
            let request_state = res.get("requestState").and_then(|r| r.as_str()).unwrap_or("").to_string();
            
            return Ok(McpToolCallOutcome::InputRequired {
                input_requests,
                request_state,
            });
        }

        let mut contents = Vec::new();
        if let Some(arr) = res.get("content").and_then(|c| c.as_array()) {
            for item in arr {
                contents.push(Self::parse_content(item));
            }
        }

        Ok(McpToolCallOutcome::Complete {
            is_error,
            content: contents,
        })
    }

    /// Parse MCP content from JSON value
    fn parse_content(value: &Value) -> McpContent {
        if let Some(content_type) = value.get("type").and_then(|t| t.as_str()) {
            match content_type {
                "text" => {
                    if let Some(text) = value.get("text").and_then(|t| t.as_str()) {
                        return McpContent::Text { text: text.to_string() };
                    }
                }
                "image" => {
                    if let (Some(data), Some(mime_type)) = (
                        value.get("data").and_then(|d| d.as_str()),
                        value.get("mimeType").and_then(|m| m.as_str()),
                    ) {
                        return McpContent::Image {
                            data: data.to_string(),
                            mime_type: mime_type.to_string(),
                        };
                    }
                }
                "resource" => {
                    if let Some(resource) = value.get("resource") {
                        return McpContent::Resource { resource: resource.clone() };
                    }
                }
                _ => {}
            }
        }
        // Unknown or unparseable content
        McpContent::Unknown
    }

    /// Validate tool arguments against JSON schema
    fn validate_basic_arguments(schema: &Value, arguments: &Value) -> Result<(), String> {
        // Basic validation - check required fields and types
        if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
            if let Some(obj) = arguments.as_object() {
                for req in required {
                    if let Some(field_name) = req.as_str() {
                        if !obj.contains_key(field_name) {
                            return Err(format!("Missing required field: {field_name}"));
                        }
                    }
                }
            } else if !required.is_empty() {
                return Err("Arguments must be an object when required fields are specified".to_string());
            }
        }

        // Check properties types if present
        if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
            if let Some(obj) = arguments.as_object() {
                for (key, val) in obj {
                    if let Some(prop_schema) = props.get(key) {
                        Self::validate_value(key, val, prop_schema)?;
                    }
                }
            }
        }

        Ok(())
    }

    fn validate_value(field: &str, value: &Value, schema: &Value) -> Result<(), String> {
        // Handle enum validation
        if let Some(enum_values) = schema.get("enum").and_then(|e| e.as_array()) {
            let matches = enum_values.iter().any(|v| v == value);
            if !matches {
                return Err(format!("Field '{field}' value must be one of: {:?}", enum_values));
            }
            return Ok(());
        }

        // Handle const validation
        if let Some(const_val) = schema.get("const") {
            if const_val != value {
                return Err(format!("Field '{field}' must be exactly: {:?}", const_val));
            }
            return Ok(());
        }

        // Handle type validation (can be string or array of strings)
        if let Some(type_val) = schema.get("type") {
            if let Some(type_str) = type_val.as_str() {
                Self::validate_single_type(field, value, type_str)?;
            } else if let Some(type_arr) = type_val.as_array() {
                let matches = type_arr.iter().any(|t| {
                    t.as_str().map_or(false, |type_str| {
                        Self::validate_single_type(field, value, type_str).is_ok()
                    })
                });
                if !matches {
                    return Err(format!("Field '{field}' must match one of types: {:?}", type_arr));
                }
            }
        }

        // Handle object properties recursively
        if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
            if let Some(obj) = value.as_object() {
                for (key, val) in obj {
                    if let Some(prop_schema) = props.get(key) {
                        Self::validate_value(&format!("{field}.{key}"), val, prop_schema)?;
                    }
                }
            } else if value != &Value::Null {
                // Check if object type is expected
                if let Some("object") = schema.get("type").and_then(|t| t.as_str()) {
                    return Err(format!("Field '{field}' expects object but got different type"));
                }
            }
        }

        // Handle array items validation
        if let Some(items_schema) = schema.get("items") {
            if let Some(arr) = value.as_array() {
                for (i, item) in arr.iter().enumerate() {
                    Self::validate_value(&format!("{field}[{i}]"), item, items_schema)?;
                }
            } else if value != &Value::Null {
                if let Some("array") = schema.get("type").and_then(|t| t.as_str()) {
                    return Err(format!("Field '{field}' expects array but got different type"));
                }
            }
        }

        // Handle string constraints
        if let Some(min_len) = schema.get("minLength").and_then(|v| v.as_u64()) {
            if let Some(s) = value.as_str() {
                if s.len() < min_len as usize {
                    return Err(format!("Field '{field}' must be at least {min_len} characters"));
                }
            }
        }
        if let Some(max_len) = schema.get("maxLength").and_then(|v| v.as_u64()) {
            if let Some(s) = value.as_str() {
                if s.len() > max_len as usize {
                    return Err(format!("Field '{field}' must be at most {max_len} characters"));
                }
            }
        }

        // Handle numeric constraints
        if let Some(minimum) = schema.get("minimum").and_then(|v| v.as_f64()) {
            if let Some(n) = value.as_f64() {
                if n < minimum {
                    return Err(format!("Field '{field}' must be at least {minimum}"));
                }
            }
        }
        if let Some(maximum) = schema.get("maximum").and_then(|v| v.as_f64()) {
            if let Some(n) = value.as_f64() {
                if n > maximum {
                    return Err(format!("Field '{field}' must be at most {maximum}"));
                }
            }
        }

        Ok(())
    }

    fn validate_single_type(field: &str, value: &Value, expected_type: &str) -> Result<(), String> {
        match (expected_type, value) {
            ("string", Value::String(_)) => Ok(()),
            ("number", Value::Number(_)) => Ok(()),
            ("integer", Value::Number(n)) if n.is_i64() || n.is_u64() => Ok(()),
            ("boolean", Value::Bool(_)) => Ok(()),
            ("object", Value::Object(_)) => Ok(()),
            ("array", Value::Array(_)) => Ok(()),
            ("null", Value::Null) => Ok(()),
            (expected, _) => Err(format!("Field '{field}' expects type '{expected}' but got different type")),
        }
    }

    /// Gracefully shut down the MCP server process
    pub async fn shutdown(&self) -> Result<(), String> {
        // Signal writer to shutdown
        let _ = self.writer_tx.send(WriterCommand::Shutdown).await;
        
        // Close stdin to signal shutdown
        {
            let mut stdin = self.stdin.lock().await;
            let _ = stdin.shutdown().await;
        }
        
        // Wait for process to exit with timeout
        let mut child = self.child.lock().await;
        match tokio::time::timeout(std::time::Duration::from_secs(5), child.wait()).await {
            Ok(Ok(status)) => {
                eprintln!("[MCP:{}] Server exited with status: {}", self.server_name, status);
                Ok(())
            }
            Ok(Err(e)) => Err(format!("Failed to wait for server process: {e}")),
            Err(_) => {
                // Force kill if timeout
                eprintln!("[MCP:{}] Server did not shut down gracefully, forcing termination", self.server_name);
                let _ = child.kill().await;
                Err("Server shutdown timed out, process killed".to_string())
            }
        }
    }
}

/// Global MCP Manager coordinating all configured MCP servers
pub struct McpManager {
    sessions: HashMap<String, Arc<Mutex<McpSession>>>,
    tools_changed_tx: mpsc::Sender<String>,
    tools_changed_rx: Arc<Mutex<mpsc::Receiver<String>>>,
}

static MCP_MANAGER: Mutex<Option<McpManager>> = Mutex::const_new(None);

impl McpManager {
    pub fn new() -> Self {
        let (tools_changed_tx, tools_changed_rx) = mpsc::channel::<String>(32);
        Self {
            sessions: HashMap::new(),
            tools_changed_tx,
            tools_changed_rx: Arc::new(Mutex::new(tools_changed_rx)),
        }
    }

    /// Returns a shared singleton instance
    pub async fn instance() -> &'static Mutex<Option<McpManager>> {
        let mut g = MCP_MANAGER.lock().await;
        if g.is_none() {
            *g = Some(McpManager::new());
        }
        &MCP_MANAGER
    }

    /// Sync active MCP sessions with current runtime settings
    pub async fn sync_with_settings(&mut self) {
        let configs = crate::settings::get_mcp_tools();
        let desired_names: Vec<String> = configs.iter().map(|c| c.name.clone()).collect();

        // Identify servers to remove (deleted in settings)
        let to_remove: Vec<String> = self.sessions
            .keys()
            .filter(|name| !desired_names.contains(name))
            .cloned()
            .collect();

        // Shutdown and remove deleted servers
        for name in to_remove {
            if let Some(session) = self.sessions.remove(&name) {
                let session = session.lock().await;
                let _ = session.shutdown().await;
            }
        }

        // Connect new or modified servers
        for config in configs {
            let should_spawn = match self.sessions.get(&config.name) {
                None => true,
                Some(_existing) => {
                    // Check if config changed (command or env)
                    // For now, always respawn if config exists to handle modifications
                    // TODO: Store config hash for proper comparison
                    true
                }
            };

            if should_spawn {
                // Shutdown existing if any
                if let Some(existing) = self.sessions.remove(&config.name) {
                    let existing = existing.lock().await;
                    let _ = existing.shutdown().await;
                }

                let env_vars = if config.env_vars.is_empty() {
                    None
                } else {
                    Some(config.env_vars.clone())
                };
                match McpSession::spawn(
                    config.name.clone(),
                    &config.command_path,
                    &config.args,
                    env_vars,
                    Some(self.tools_changed_tx.clone()),
                ).await {
                    Ok(sess) => {
                        self.sessions.insert(config.name.clone(), Arc::new(Mutex::new(sess)));
                    }
                    Err(e) => {
                        eprintln!("[MCP] Failed to connect server '{}': {}", config.name, e);
                    }
                }
            }
        }
    }

    /// Shutdown all MCP sessions gracefully
    pub async fn shutdown_all(&mut self) {
        for (_name, session) in self.sessions.drain() {
            let session = session.lock().await;
            let _ = session.shutdown().await;
        }
    }

    /// Process any pending tools_changed notifications
    pub async fn process_tools_changed(&mut self) {
        let mut rx = self.tools_changed_rx.lock().await;
        while let Ok(server_name) = rx.try_recv() {
            if let Some(session) = self.sessions.get(&server_name) {
                let mut session = session.lock().await;
                let _ = session.check_and_refresh_tools().await;
            }
        }
    }

    /// Get a sender for tools_changed notifications
    pub fn get_tools_changed_sender(&self) -> mpsc::Sender<String> {
        self.tools_changed_tx.clone()
    }

    /// Formats all available MCP tools across all active servers for prompt injection
    pub async fn generate_prompt_tools_section(&self) -> String {
        if self.sessions.is_empty() {
            return String::new();
        }

        let mut out = String::from("\n## Active Model Context Protocol (MCP) Tools\n");
        out.push_str("To invoke an MCP tool, output:\n");
        out.push_str("<mcp server=\"SERVER_NAME\" tool=\"TOOL_NAME\">\n");
        out.push_str("{\n  \"param_name\": \"value\"\n}\n");
        out.push_str("</mcp>\n\n");
        out.push_str("Available MCP Tools:\n");

        for (server_name, sess) in &self.sessions {
            let sess = sess.lock().await;
            out.push_str(&format!("- Server `{server_name}`:\n"));
            if sess.tools.is_empty() {
                out.push_str("  (No tools discovered on this server)\n");
            } else {
                for t in &sess.tools {
                    let desc = t.description.as_deref().unwrap_or("No description");
                    let schema_str = serde_json::to_string(&t.input_schema).unwrap_or_default();
                    out.push_str(&format!("  * `{}`: {} | Schema: {}\n", t.name, desc, schema_str));
                }
            }
        }

        out
    }

    /// Call an MCP tool across the registered sessions
    pub async fn execute_tool(
        &self,
        server_hint: Option<&str>,
        tool_name: &str,
        arguments_json: &str,
    ) -> Result<McpToolCallOutcome, String> {
        let args_val: Value = if arguments_json.trim().is_empty() {
            json!({})
        } else {
            match serde_json::from_str(arguments_json) {
                Ok(v) => v,
                Err(e) => return Err(format!("Error parsing MCP tool arguments JSON: {e}")),
            }
        };

        // If server is specified, use it directly
        if let Some(srv) = server_hint {
            if let Some(sess) = self.sessions.get(srv) {
                let sess = sess.lock().await;
                return sess.call_tool(tool_name, args_val, None, None).await
                    .map_err(|e| format!("MCP Error from server '{srv}': {e}"));
            }
        }

        // Otherwise, search all active sessions for this tool name
        for (srv_name, sess) in &self.sessions {
            let sess = sess.lock().await;
            if sess.tools.iter().any(|t| t.name == tool_name) {
                return sess.call_tool(tool_name, args_val, None, None).await
                    .map_err(|e| format!("MCP Error from server '{srv_name}': {e}"));
            }
        }

        Err(format!("Error: MCP tool '{tool_name}' not found on any active MCP server."))
    }
}

/// Simple shell words splitter without invoking a shell
fn shell_words_split(input: &str) -> Result<Vec<String>, String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut escape = false;

    for c in input.chars() {
        if escape {
            cur.push(c);
            escape = false;
            continue;
        }
        if c == '\\' && !in_single {
            escape = true;
            continue;
        }
        if c == '\'' && !in_double {
            in_single = !in_single;
            continue;
        }
        if c == '"' && !in_single {
            in_double = !in_double;
            continue;
        }
        if c.is_whitespace() && !in_single && !in_double {
            if !cur.is_empty() {
                words.push(cur.clone());
                cur.clear();
            }
        } else {
            cur.push(c);
        }
    }

    if in_single || in_double {
        return Err("Unclosed quote in command line".to_string());
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    Ok(words)
}
